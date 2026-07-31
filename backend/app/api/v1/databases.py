import uuid
from typing import List, Optional
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.core.dependencies import require_admin
from app.models.user import User
from app.models.role import Role
from app.models.department import Department
from app.models.external_database import (
    ExternalDatabaseConnection,
    DatabaseSchemaCache,
    DatabaseAccessPolicy,
)
from app.schemas.database import (
    DatabaseConnectionTestRequest,
    DatabaseConnectionCreate,
    DatabaseConnectionUpdate,
    DatabaseConnectionResponse,
    DatabaseAccessPolicyCreate,
    DatabaseAccessPolicyResponse,
)
from app.services.database_service import (
    test_connection_live,
    introspect_schema_live,
    encrypt_password,
    decrypt_password,
    check_user_db_access,
    get_user_filtered_schema,
)
from app.services.role_service import get_role_ancestors
from app.services.audit_log_service import log_audit_event
from app.core.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/test-connection", status_code=status.HTTP_200_OK)
def test_connection(
    request: DatabaseConnectionTestRequest,
    current_admin: User = Depends(require_admin),
):
    """
    Test a database connection configuration before creating it.
    """
    try:
        test_connection_live(
            engine_type=request.engine,
            host=request.host,
            port=request.port,
            database_name=request.database_name,
            username=request.username,
            password_decrypted=request.password,
            ssl_mode=request.ssl_mode,
        )
        return {"status": "success", "message": "Connection tested successfully"}
    except Exception as e:
        logger.error("Connection test failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Connection test failed: {str(e)}",
        )


@router.post(
    "", response_model=DatabaseConnectionResponse, status_code=status.HTTP_201_CREATED
)
def create_database_connection(
    request: DatabaseConnectionCreate,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Create a new external database connection.
    Tests the connection first, encrypts the password, reflects the schema, and saves the connection.
    """
    # 1. Test the connection first
    try:
        print("TESTING")
        test_connection_live(
            engine_type=request.engine,
            host=request.host,
            port=request.port,
            database_name=request.database_name,
            username=request.username,
            password_decrypted=request.password,
            ssl_mode=request.ssl_mode,
        )
        print("TESTED")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to test connection: {str(e)}",
        )

    # 2. Introspect schema
    try:
        print("INTROSPECTING")
        schema_data = introspect_schema_live(
            engine_type=request.engine,
            host=request.host,
            port=request.port,
            database_name=request.database_name,
            username=request.username,
            password_decrypted=request.password,
            ssl_mode=request.ssl_mode,
        )
        print("INTROSPECTED")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to introspect schema: {str(e)}",
        )

    # 3. Encrypt password and save connection
    encrypted_pw = encrypt_password(request.password)

    print("SAVING CONNECTION")
    conn = ExternalDatabaseConnection(
        tenant_id=current_admin.tenant_id,
        name=request.name,
        engine=request.engine,
        host=request.host,
        port=request.port,
        database_name=request.database_name,
        username=request.username,
        password=encrypted_pw,
        ssl_mode=request.ssl_mode,
        status="active",
        last_synced_at=datetime.now(timezone.utc),
    )
    db.add(conn)
    db.flush()  # gets conn.id
    print("CONNECTION SAVED")

    # 4. Save introspected schema cache
    schema_cache = DatabaseSchemaCache(
        connection_id=conn.id,
        schema_data=schema_data,
    )
    db.add(schema_cache)
    print("SCHEMA CACHE SAVED")

    # 5. Log audit
    log_audit_event(
        db=db,
        tenant_id=current_admin.tenant_id,
        actor_user_id=current_admin.id,
        action="CREATE_DATABASE_CONNECTION",
        description=f"Created database connection {request.name} ({request.engine})",
    )
    print("AUDIT LOGGED")
    db.commit()
    db.refresh(conn)
    return conn


@router.get("/authorized", response_model=List[DatabaseConnectionResponse])
def list_authorized_database_connections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all external database connections the current user is authorized to query.
    """

    connections = (
        db.query(ExternalDatabaseConnection)
        .options(joinedload(ExternalDatabaseConnection.schema_cache))
        .filter(ExternalDatabaseConnection.tenant_id == current_user.tenant_id)
        .all()
    )

    authorized_connections = []
    for conn in connections:
        if check_user_db_access(db, current_user, conn.id):
            authorized_connections.append(conn)

    return authorized_connections


@router.get("", response_model=List[DatabaseConnectionResponse])
def list_database_connections(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    List all external database connections for the tenant.
    """
    connections = (
        db.query(ExternalDatabaseConnection)
        .options(joinedload(ExternalDatabaseConnection.schema_cache))
        .filter(ExternalDatabaseConnection.tenant_id == current_admin.tenant_id)
        .all()
    )
    return connections


@router.get("/{id}", response_model=DatabaseConnectionResponse)
def get_database_connection(
    id: uuid.UUID,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Get connection details.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .options(joinedload(ExternalDatabaseConnection.schema_cache))
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )
    return conn


@router.get("/{id}/schema")
def get_database_schema(
    id: uuid.UUID,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Get the cached schema JSON of the database connection.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    cache = (
        db.query(DatabaseSchemaCache)
        .filter(DatabaseSchemaCache.connection_id == id)
        .first()
    )
    return cache.schema_data if cache else {"tables": []}


@router.get("/{id}/user-schema")
def get_database_user_schema(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the filtered cached schema JSON of the database connection for the calling user.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_user.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    cache = (
        db.query(DatabaseSchemaCache)
        .filter(DatabaseSchemaCache.connection_id == id)
        .first()
    )
    raw_schema = cache.schema_data if cache else {"tables": []}
    return get_user_filtered_schema(db, current_user, id, raw_schema)


@router.put("/{id}", response_model=DatabaseConnectionResponse)
def update_database_connection(
    id: uuid.UUID,
    request: DatabaseConnectionUpdate,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Update connection parameters.
    Re-tests the connection if parameters change.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    # Determine if credentials/params changed
    test_needed = False

    if request.name is not None:
        conn.name = request.name
    if request.host is not None and request.host != conn.host:
        conn.host = request.host
        test_needed = True
    if request.port is not None and request.port != conn.port:
        conn.port = request.port
        test_needed = True
    if (
        request.database_name is not None
        and request.database_name != conn.database_name
    ):
        conn.database_name = request.database_name
        test_needed = True
    if request.username is not None and request.username != conn.username:
        conn.username = request.username
        test_needed = True
    if request.password is not None:
        conn.password = encrypt_password(request.password)
        test_needed = True
    if request.ssl_mode is not None and request.ssl_mode != conn.ssl_mode:
        conn.ssl_mode = request.ssl_mode
        test_needed = True

    if test_needed:
        # Run test connection with the updated state
        pwd_to_test = decrypt_password(conn.password)
        try:
            test_connection_live(
                engine_type=conn.engine,
                host=conn.host,
                port=conn.port,
                database_name=conn.database_name,
                username=conn.username,
                password_decrypted=pwd_to_test,
                ssl_mode=conn.ssl_mode,
            )
            # Re-sync schema since credentials/host changed
            schema_data = introspect_schema_live(
                engine_type=conn.engine,
                host=conn.host,
                port=conn.port,
                database_name=conn.database_name,
                username=conn.username,
                password_decrypted=pwd_to_test,
                ssl_mode=conn.ssl_mode,
            )
            cache = (
                db.query(DatabaseSchemaCache)
                .filter(DatabaseSchemaCache.connection_id == id)
                .first()
            )
            if cache:
                cache.schema_data = schema_data
            else:
                db.add(DatabaseSchemaCache(connection_id=id, schema_data=schema_data))

            conn.status = "active"
            conn.last_error = None
            conn.last_synced_at = datetime.now(timezone.utc)
        except Exception as e:
            conn.status = "error"
            conn.last_error = str(e)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Updated parameters failed connection test: {str(e)}",
            )

    log_audit_event(
        db=db,
        tenant_id=current_admin.tenant_id,
        actor_user_id=current_admin.id,
        action="UPDATE_DATABASE_CONNECTION",
        description=f"Updated database connection {conn.name}",
    )
    db.commit()
    db.refresh(conn)
    return conn


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_database_connection(
    id: uuid.UUID,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Delete a database connection. Automatically cascade deletes cache and policies.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    db.delete(conn)
    log_audit_event(
        db=db,
        tenant_id=current_admin.tenant_id,
        actor_user_id=current_admin.id,
        action="DELETE_DATABASE_CONNECTION",
        description=f"Deleted database connection {conn.name}",
    )
    db.commit()
    return


@router.post("/{id}/refresh", response_model=DatabaseConnectionResponse)
def refresh_database_schema(
    id: uuid.UUID,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Manually refreshes the schema from the live database and updates the schema cache.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    try:
        pwd_decrypted = decrypt_password(conn.password)
        schema_data = introspect_schema_live(
            engine_type=conn.engine,
            host=conn.host,
            port=conn.port,
            database_name=conn.database_name,
            username=conn.username,
            password_decrypted=pwd_decrypted,
            ssl_mode=conn.ssl_mode,
        )
        cache = (
            db.query(DatabaseSchemaCache)
            .filter(DatabaseSchemaCache.connection_id == id)
            .first()
        )
        if cache:
            cache.schema_data = schema_data
        else:
            db.add(DatabaseSchemaCache(connection_id=id, schema_data=schema_data))

        conn.status = "active"
        conn.last_error = None
        conn.last_synced_at = datetime.now(timezone.utc)
    except Exception as e:
        conn.status = "error"
        conn.last_error = str(e)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Schema refresh failed: {str(e)}",
        )

    log_audit_event(
        db=db,
        tenant_id=current_admin.tenant_id,
        actor_user_id=current_admin.id,
        action="REFRESH_DATABASE_SCHEMA",
        description=f"Refreshed schema for {conn.name}",
    )
    db.commit()
    db.refresh(conn)
    return conn


# --- Access Policy Endpoints ---


@router.get("/{id}/access", response_model=List[DatabaseAccessPolicyResponse])
def list_connection_access_policies(
    id: uuid.UUID,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    List access policies on a database connection.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    policies = (
        db.query(DatabaseAccessPolicy)
        .filter(DatabaseAccessPolicy.connection_id == id)
        .options(
            joinedload(DatabaseAccessPolicy.role),
            joinedload(DatabaseAccessPolicy.inherited_from_role),
            joinedload(DatabaseAccessPolicy.granted_via_department),
        )
        .all()
    )

    # Format output for response model (attaching resolved names)
    response_list = []
    for policy in policies:
        response_list.append(
            DatabaseAccessPolicyResponse(
                id=policy.id,
                connection_id=policy.connection_id,
                role_id=policy.role_id,
                role_name=policy.role.name,
                granted_via=policy.granted_via,
                inherited_from_role_id=policy.inherited_from_role_id,
                inherited_from_role_name=policy.inherited_from_role.name
                if policy.inherited_from_role
                else None,
                granted_via_department_id=policy.granted_via_department_id,
                granted_via_department_name=policy.granted_via_department.name
                if policy.granted_via_department
                else None,
                table_name=policy.table_name,
                columns=policy.columns,
                created_at=policy.created_at,
            )
        )
    return response_list


def filter_columns_for_table(
    columns: Optional[List[str]], table_name: Optional[str]
) -> Optional[List[str]]:
    if not columns:
        return None
    if not table_name:
        return columns

    filtered = []
    for col_spec in columns:
        if "." in col_spec:
            parts = col_spec.split(".", 1)
            if parts[0].lower() == table_name.lower():
                filtered.append(parts[1])
        else:
            filtered.append(col_spec)
    return filtered


@router.post("/{id}/access", response_model=List[DatabaseAccessPolicyResponse])
def assign_connection_access(
    id: uuid.UUID,
    request: DatabaseAccessPolicyCreate,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Grants access connection to a role or department, optionally scoped to a table.
    Automatically generates inherited policies matching documents RBAC hierarchy rules.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    target_roles = []
    if request.role_ids:
        target_roles = request.role_ids
    elif request.role_id:
        target_roles = [request.role_id]

    target_departments = []
    if request.department_ids:
        target_departments = request.department_ids
    elif request.department_id:
        target_departments = [request.department_id]

    if not target_roles and not target_departments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either role_ids or department_ids must be provided",
        )

    target_tables = []
    if request.table_names is not None:
        target_tables = request.table_names
    else:
        target_tables = [request.table_name]

    if request.table_names is not None and not request.table_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one table must be selected",
        )

    added_policies = []

    # 1. Process Role-Based Grant
    if target_roles:
        from app.services.notification_service import create_notification
        from app.models.enums import NotificationType

        for r_id in target_roles:
            role = (
                db.query(Role)
                .filter(Role.id == r_id, Role.tenant_id == current_admin.tenant_id)
                .first()
            )
            if not role:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Role with ID {r_id} not found",
                )

            for tbl in target_tables:
                tbl_columns = filter_columns_for_table(request.columns, tbl)

                # Create direct policy if not already existing
                existing = (
                    db.query(DatabaseAccessPolicy)
                    .filter(
                        DatabaseAccessPolicy.connection_id == id,
                        DatabaseAccessPolicy.role_id == r_id,
                        DatabaseAccessPolicy.table_name == tbl,
                    )
                    .first()
                )
                if not existing:
                    direct_policy = DatabaseAccessPolicy(
                        connection_id=id,
                        role_id=r_id,
                        granted_via="direct",
                        table_name=tbl,
                        columns=tbl_columns,
                    )
                    db.add(direct_policy)
                    added_policies.append(direct_policy)

                    # Notify direct role users
                    role_users = db.query(User).filter(User.role_id == r_id).all()
                    table_info = f" (table: {tbl})" if tbl else ""
                    for user in role_users:
                        create_notification(
                            db=db,
                            user_id=user.id,
                            tenant_id=user.tenant_id,
                            type=NotificationType.database_access_direct,
                            message=f"You have been granted direct access to database: {conn.name}{table_info}",
                            related_role_id=r_id,
                            flush_only=True,
                        )

                # Walk up role hierarchy for upward inheritance
                ancestors = get_role_ancestors(r_id, db)
                for ancestor in ancestors:
                    # Check if this ancestor already has an access policy
                    existing_anc = (
                        db.query(DatabaseAccessPolicy)
                        .filter(
                            DatabaseAccessPolicy.connection_id == id,
                            DatabaseAccessPolicy.role_id == ancestor.id,
                            DatabaseAccessPolicy.table_name == tbl,
                        )
                        .first()
                    )
                    if not existing_anc:
                        anc_policy = DatabaseAccessPolicy(
                            connection_id=id,
                            role_id=ancestor.id,
                            granted_via="inherited",
                            inherited_from_role_id=r_id,
                            table_name=tbl,
                            columns=tbl_columns,
                        )
                        db.add(anc_policy)
                        added_policies.append(anc_policy)

                        # Notify ancestor role users
                        ancestor_users = (
                            db.query(User).filter(User.role_id == ancestor.id).all()
                        )
                        table_info = f" (table: {tbl})" if tbl else ""
                        for user in ancestor_users:
                            create_notification(
                                db=db,
                                user_id=user.id,
                                tenant_id=user.tenant_id,
                                type=NotificationType.database_access_inherited_hierarchy,
                                message=f"You have been granted inherited access to database: {conn.name}{table_info} (inherited from role: {role.name})",
                                related_role_id=ancestor.id,
                                flush_only=True,
                            )

    # 2. Process Department-Based Grant (no hierarchy inheritance)
    if target_departments:
        from app.services.notification_service import create_notification
        from app.models.enums import NotificationType

        for dept_id in target_departments:
            dept = (
                db.query(Department)
                .filter(
                    Department.id == dept_id,
                    Department.tenant_id == current_admin.tenant_id,
                )
                .first()
            )
            if not dept:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Department with ID {dept_id} not found",
                )

            # Find all roles assigned to this department
            roles = (
                db.query(Role)
                .filter(
                    Role.department_id == dept_id,
                    Role.tenant_id == current_admin.tenant_id,
                )
                .all()
            )
            for role in roles:
                for tbl in target_tables:
                    tbl_columns = filter_columns_for_table(request.columns, tbl)

                    existing = (
                        db.query(DatabaseAccessPolicy)
                        .filter(
                            DatabaseAccessPolicy.connection_id == id,
                            DatabaseAccessPolicy.role_id == role.id,
                            DatabaseAccessPolicy.table_name == tbl,
                        )
                        .first()
                    )
                    if not existing:
                        dept_policy = DatabaseAccessPolicy(
                            connection_id=id,
                            role_id=role.id,
                            granted_via="department",
                            granted_via_department_id=dept_id,
                            table_name=tbl,
                            columns=tbl_columns,
                        )
                        db.add(dept_policy)
                        added_policies.append(dept_policy)

            # Notify users whose role belongs to that department
            dept_users = (
                db.query(User).join(Role).filter(Role.department_id == dept_id).all()
            )
            tables_str = (
                ", ".join([t for t in target_tables if t])
                if any(target_tables)
                else "all"
            )
            table_info = f" (tables: {tables_str})" if tables_str != "all" else ""
            for u in dept_users:
                create_notification(
                    db=db,
                    user_id=u.id,
                    tenant_id=u.tenant_id,
                    type=NotificationType.database_access_inherited_department,
                    message=f"You have been granted access to database: {conn.name}{table_info} (via department: {dept.name})",
                    related_department_id=dept_id,
                    flush_only=True,
                )

    db.commit()

    # Re-fetch populated policies to return
    response_list = []
    for policy in added_policies:
        db.refresh(policy)
        response_list.append(
            DatabaseAccessPolicyResponse(
                id=policy.id,
                connection_id=policy.connection_id,
                role_id=policy.role_id,
                role_name=policy.role.name,
                granted_via=policy.granted_via,
                inherited_from_role_id=policy.inherited_from_role_id,
                inherited_from_role_name=policy.inherited_from_role.name
                if policy.inherited_from_role
                else None,
                granted_via_department_id=policy.granted_via_department_id,
                granted_via_department_name=policy.granted_via_department.name
                if policy.granted_via_department
                else None,
                table_name=policy.table_name,
                columns=policy.columns,
                created_at=policy.created_at,
            )
        )

    log_audit_event(
        db=db,
        tenant_id=current_admin.tenant_id,
        actor_user_id=current_admin.id,
        action="GRANT_DATABASE_ACCESS",
        description=f"Granted access to database connection {conn.name}",
    )
    return response_list


@router.delete("/{id}/access/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_connection_access(
    id: uuid.UUID,
    policy_id: uuid.UUID,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Revokes database access by deleting the direct policy and cleaning up all associated inherited/department policies.
    """
    conn = (
        db.query(ExternalDatabaseConnection)
        .filter(
            ExternalDatabaseConnection.id == id,
            ExternalDatabaseConnection.tenant_id == current_admin.tenant_id,
        )
        .first()
    )
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found",
        )

    policy = (
        db.query(DatabaseAccessPolicy)
        .filter(
            DatabaseAccessPolicy.id == policy_id,
            DatabaseAccessPolicy.connection_id == id,
        )
        .first()
    )
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access policy not found",
        )

    # If it is a direct role grant, we also cascade-delete policies inherited from this role
    if policy.granted_via == "direct":
        inherited_policies = (
            db.query(DatabaseAccessPolicy)
            .filter(
                DatabaseAccessPolicy.connection_id == id,
                DatabaseAccessPolicy.inherited_from_role_id == policy.role_id,
                DatabaseAccessPolicy.table_name == policy.table_name,
            )
            .all()
        )
        for ip in inherited_policies:
            db.delete(ip)

    db.delete(policy)
    log_audit_event(
        db=db,
        tenant_id=current_admin.tenant_id,
        actor_user_id=current_admin.id,
        action="REVOKE_DATABASE_ACCESS",
        description=f"Revoked database connection access policy {policy_id}",
    )
    db.commit()
    return
