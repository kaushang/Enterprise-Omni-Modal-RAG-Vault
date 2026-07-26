import sys
import os
import uuid
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base
from app.models.role import Role
from app.models.user import User
from app.models.department import Department
from app.models.external_database import (
    ExternalDatabaseConnection,
    DatabaseAccessPolicy,
)
from app.services.database_service import (
    encrypt_password,
    decrypt_password,
    check_user_db_access,
    get_user_authorized_tables,
    get_connection_url,
    run_query_on_connection,
)

DATABASE_URL = "sqlite:///:memory:"


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"


@pytest.fixture(name="db")
def db_fixture():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_encryption_decryption():
    """
    Test password symmetric encryption and decryption.
    """
    original = "super-secret-password-123"
    encrypted = encrypt_password(original)
    assert encrypted != original

    decrypted = decrypt_password(encrypted)
    assert decrypted == original


def test_connection_url_generation():
    """
    Test URL construction based on engine type.
    """
    # Postgres
    url_pg = get_connection_url(
        engine_type="postgresql",
        host="127.0.0.1",
        port=5432,
        database_name="mydb",
        username="postgres",
        password_decrypted="password",
        ssl_mode="require",
    )
    assert "postgresql://" in url_pg
    assert "sslmode=require" in url_pg

    # MySQL
    url_my = get_connection_url(
        engine_type="mysql",
        host="localhost",
        port=3306,
        database_name="mydb",
        username="user",
        password_decrypted="pwd",
        ssl_mode="require",
    )
    assert "mysql+pymysql://" in url_my
    assert "ssl=true" in url_my


def test_access_policy_role_inheritance(db):
    """
    Verify check_user_db_access resolves role inheritance upward.
    """
    tenant_id = uuid.uuid4()

    # Setup Roles hierarchy: RootRole -> ManagerRole -> StaffRole
    root_role = Role(
        id=uuid.uuid4(), tenant_id=tenant_id, name="Director", is_admin=False
    )
    manager_role = Role(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Manager",
        parent_role_id=root_role.id,
        is_admin=False,
    )
    staff_role = Role(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Staff",
        parent_role_id=manager_role.id,
        is_admin=False,
    )

    db.add_all([root_role, manager_role, staff_role])
    db.commit()

    conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Main DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="production",
        username="user",
        password="encrypted",
        status="active",
    )
    db.add(conn)
    db.commit()

    # Assign access to ManagerRole
    policy = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=conn.id,
        role_id=manager_role.id,
        granted_via="direct",
    )
    # Inheritance: Director inherits access
    inherited_policy = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=conn.id,
        role_id=root_role.id,
        granted_via="inherited",
        inherited_from_role_id=manager_role.id,
    )
    db.add_all([policy, inherited_policy])
    db.commit()

    # Create users
    director_user = User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email="d@test.com",
        hashed_password="pwd",
        full_name="Director",
        role_id=root_role.id,
        role=root_role,
    )
    manager_user = User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email="m@test.com",
        hashed_password="pwd",
        full_name="Manager",
        role_id=manager_role.id,
        role=manager_role,
    )
    staff_user = User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email="s@test.com",
        hashed_password="pwd",
        full_name="Staff",
        role_id=staff_role.id,
        role=staff_role,
    )

    # 1. Manager has access (direct)
    assert check_user_db_access(db, manager_user, conn.id) is True

    # 2. Director has access (inherited)
    assert check_user_db_access(db, director_user, conn.id) is True

    # 3. Staff does NOT have access
    assert check_user_db_access(db, staff_user, conn.id) is False


def test_get_user_authorized_tables(db):
    """
    Verify get_user_authorized_tables returns whole db or specific tables.
    """
    tenant_id = uuid.uuid4()
    role = Role(id=uuid.uuid4(), tenant_id=tenant_id, name="Staff", is_admin=False)
    user = User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email="u@test.com",
        hashed_password="pwd",
        full_name="Staff User",
        role_id=role.id,
        role=role,
    )
    db.add_all([role, user])
    db.commit()

    conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Main DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="production",
        username="user",
        password="encrypted",
        status="active",
    )
    db.add(conn)
    db.commit()

    all_tables = ["users", "orders", "payments", "logs"]

    # 1. Initially, no access
    assert get_user_authorized_tables(db, user, conn.id, all_tables) == []

    # 2. Add table-level policy
    policy_orders = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=conn.id,
        role_id=role.id,
        granted_via="direct",
        table_name="orders",
    )
    db.add(policy_orders)
    db.commit()

    assert get_user_authorized_tables(db, user, conn.id, all_tables) == ["orders"]

    # 3. Add database-level policy (table_name is NULL)
    policy_all = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=conn.id,
        role_id=role.id,
        granted_via="direct",
        table_name=None,
    )
    db.add(policy_all)
    db.commit()

    # Should return all tables now
    assert set(get_user_authorized_tables(db, user, conn.id, all_tables)) == set(
        all_tables
    )


def test_schema_drift_detection():
    """
    Verify connection query execution raises schema drift validation error
    when missing columns or tables are encountered.
    """
    conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        name="Main DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="production",
        username="user",
        password=encrypt_password("pwd"),
        status="active",
    )

    schema_cache_tables = [
        {
            "name": "users",
            "columns": [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"},
            ],
        }
    ]

    # Mock the SQLAlchemy engine and connection execution to raise UndefinedColumn
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    from sqlalchemy.exc import ProgrammingError

    def mock_execute(statement, *args, **kwargs):
        sql = str(statement).lower()
        if "set transaction" in sql:
            return MagicMock()
        raise ProgrammingError(sql, {}, 'relation "users" does not exist')

    mock_conn.execute.side_effect = mock_execute

    with patch("app.services.database_service.create_engine", return_value=mock_engine):
        with pytest.raises(ValueError) as excinfo:
            run_query_on_connection(
                conn, "SELECT name, age FROM users", schema_cache_tables
            )

        assert "schema mismatch (drift)" in str(excinfo.value)
        assert "Refresh Schema" in str(excinfo.value)


def test_database_citations_storage(db):
    """
    Verify QueryCitation successfully supports connection_id and nullable document_id.
    """
    from app.models.query_session import QuerySession
    from app.models.query_message import QueryMessage
    from app.models.query_citation import QueryCitation
    from app.models.enums import MessageRole

    tenant_id = uuid.uuid4()
    role = Role(id=uuid.uuid4(), tenant_id=tenant_id, name="Staff", is_admin=False)
    user = User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email="test_cit@test.com",
        hashed_password="pwd",
        full_name="User",
        role_id=role.id,
        role=role,
    )
    db.add_all([role, user])
    db.commit()

    conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Citation DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="production",
        username="user",
        password="encrypted",
        status="active",
    )
    db.add(conn)
    db.commit()

    session = QuerySession(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        user_id=user.id,
        title="Test Session",
    )
    db.add(session)
    db.commit()

    message = QueryMessage(
        id=uuid.uuid4(),
        session_id=session.id,
        role=MessageRole.assistant,
        content="Here is your database result.",
    )
    db.add(message)
    db.commit()

    # Create citation pointing to connection_id with document_id set to None
    citation = QueryCitation(
        id=uuid.uuid4(),
        message_id=message.id,
        document_id=None,
        connection_id=conn.id,
        qdrant_vector_id="0",
        chunk_text="SQL: SELECT * FROM users;\nResults: []",
        page_number=None,
        chunk_index=0,
    )
    db.add(citation)
    db.commit()

    # Query back to verify persistence and relationships
    saved_citation = (
        db.query(QueryCitation).filter(QueryCitation.id == citation.id).first()
    )
    assert saved_citation is not None
    assert saved_citation.document_id is None
    assert saved_citation.connection_id == conn.id
    assert saved_citation.connection.name == "Citation DB"


def test_revoke_department_access_member_only(db):
    """
    Verify revoking database access for a specific department member deletes
    only that member's policy and does not cascade to other department members.
    """
    tenant_id = uuid.uuid4()
    dept = Department(id=uuid.uuid4(), tenant_id=tenant_id, name="HR")

    role1 = Role(
        id=uuid.uuid4(), tenant_id=tenant_id, name="HR Recruiter", department_id=dept.id
    )
    role2 = Role(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="HR Generalist",
        department_id=dept.id,
    )

    db.add_all([dept, role1, role2])
    db.commit()

    conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Org DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="production",
        username="user",
        password="encrypted",
        status="active",
    )
    db.add(conn)
    db.commit()

    # Create policies for both roles under the department grant
    policy1 = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=conn.id,
        role_id=role1.id,
        granted_via="department",
        granted_via_department_id=dept.id,
    )
    policy2 = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=conn.id,
        role_id=role2.id,
        granted_via="department",
        granted_via_department_id=dept.id,
    )
    db.add_all([policy1, policy2])
    db.commit()

    # Verify both policies exist
    assert db.query(DatabaseAccessPolicy).count() == 2

    # Delete policy1 only
    db.delete(policy1)
    db.commit()

    # Verify policy1 is deleted, but policy2 still remains
    assert (
        db.query(DatabaseAccessPolicy)
        .filter(DatabaseAccessPolicy.id == policy1.id)
        .first()
        is None
    )
    assert (
        db.query(DatabaseAccessPolicy)
        .filter(DatabaseAccessPolicy.id == policy2.id)
        .first()
        is not None
    )
    assert db.query(DatabaseAccessPolicy).count() == 1


def test_is_value_mismatch_error():
    from app.services.rag_service import is_value_mismatch_error

    # Postgres 22P02 exception simulation
    class PostgresMockError(Exception):
        pass

    pg_err = PostgresMockError("invalid input value for enum visibility: 'public'")

    class MockOrigPg:
        pgcode = "22P02"

    pg_err.orig = MockOrigPg()
    assert is_value_mismatch_error("postgresql", pg_err) is True

    # Fallback postgres string check
    pg_err_fallback = Exception("invalid input value for enum visibility: 'public'")
    assert is_value_mismatch_error("postgresql", pg_err_fallback) is True

    # MySQL 1265 (Data truncated) error simulation
    class MySQLMockError(Exception):
        pass

    mysql_err = MySQLMockError("Data truncated for column 'visibility' at row 1")

    class MockOrigMysql:
        args = (1265, "Data truncated")

    mysql_err.orig = MockOrigMysql()
    assert is_value_mismatch_error("mysql", mysql_err) is True

    # Fallback mysql string check
    mysql_err_fallback = Exception("data truncated for column")
    assert is_value_mismatch_error("mysql", mysql_err_fallback) is True

    # Generic error should be False
    other_err = Exception("Connection refused")
    assert is_value_mismatch_error("postgresql", other_err) is False
