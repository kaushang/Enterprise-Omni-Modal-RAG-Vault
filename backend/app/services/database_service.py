import base64
import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from anthropic import AsyncAnthropic
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.enums import DatabaseEngine
from app.models.external_database import (
    DatabaseAccessPolicy,
    ExternalDatabaseConnection,
)
from app.models.user import User

logger = logging.getLogger(__name__)


# --- Model ---
_async_anthropic_client: AsyncAnthropic | None = None


def _get_async_anthropic_client() -> AsyncAnthropic:
    """Lazily initialise and return the AsyncAnthropic client."""
    global _async_anthropic_client
    if _async_anthropic_client is None:
        _async_anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _async_anthropic_client


# --- Encryption Helpers ---


def _get_encryption_key() -> bytes:
    """
    Get or derive the 32-byte URL-safe base64 encryption key.
    """
    if settings.DATABASE_ENCRYPTION_KEY:
        try:
            # Check if it's already a valid Fernet key
            key = settings.DATABASE_ENCRYPTION_KEY.encode()
            Fernet(key)
            return key
        except Exception:
            pass

    # Derive key from SECRET_KEY
    salt = b"rag_vault_salt_db_enc"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    derived = kdf.derive(settings.SECRET_KEY.encode())
    return base64.urlsafe_b64encode(derived)


def encrypt_password(password: str) -> str:
    """
    Encrypt a plaintext password using Fernet symmetric encryption.
    """
    key = _get_encryption_key()
    fernet = Fernet(key)
    return fernet.encrypt(password.encode()).decode()


def decrypt_password(encrypted_password: str) -> str:
    """
    Decrypt an encrypted password using Fernet symmetric encryption.
    """
    key = _get_encryption_key()
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_password.encode()).decode()


# --- Connection Helpers ---


def get_connection_url(
    engine_type: str,
    host: str,
    port: int,
    database_name: str,
    username: str,
    password_decrypted: str,
    ssl_mode: Optional[str] = None,
) -> str:
    """
    Constructs the SQLAlchemy connection URL based on engine and parameters.
    """
    from urllib.parse import quote_plus

    pwd_escaped = quote_plus(password_decrypted)
    user_escaped = quote_plus(username)
    host_escaped = quote_plus(host)
    db_escaped = quote_plus(database_name)

    if engine_type == DatabaseEngine.postgresql:
        ssl_part = f"?sslmode={ssl_mode}" if ssl_mode else ""
        return f"postgresql://{user_escaped}:{pwd_escaped}@{host_escaped}:{port}/{db_escaped}{ssl_part}"
    elif engine_type == DatabaseEngine.mysql:
        # SSL mode mappings for PyMySQL
        ssl_part = ""
        if ssl_mode:
            if ssl_mode.lower() in ("require", "required"):
                ssl_part = "?ssl=true"
            elif ssl_mode.lower() in ("verify-ca", "verify-full"):
                ssl_part = "?ssl_verify_cert=true"
        return f"mysql+pymysql://{user_escaped}:{pwd_escaped}@{host_escaped}:{port}/{db_escaped}{ssl_part}"
    else:
        raise ValueError(f"Unsupported database engine: {engine_type}")


def test_connection_live(
    engine_type: str,
    host: str,
    port: int,
    database_name: str,
    username: str,
    password_decrypted: str,
    ssl_mode: Optional[str] = None,
) -> None:
    """
    Tests reachability and credentials by running a live connection and selecting 1.
    """
    url = get_connection_url(
        engine_type=engine_type,
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password_decrypted=password_decrypted,
        ssl_mode=ssl_mode,
    )

    # Fast timeouts for interactive testing
    connect_args = {}
    if engine_type == DatabaseEngine.postgresql:
        connect_args = {"connect_timeout": 5}
    elif engine_type == DatabaseEngine.mysql:
        connect_args = {"connect_timeout": 5}

    engine = create_engine(url, connect_args=connect_args)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        engine.dispose()


# --- Introspection Helpers ---


def introspect_schema_live(
    engine_type: str,
    host: str,
    port: int,
    database_name: str,
    username: str,
    password_decrypted: str,
    ssl_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Connects to the database and reflects its schema.
    Returns:
        Dict representation of tables, columns, PKs, and FKs.
    """

    print("=" * 80)
    print("[START] Starting schema introspection")
    print(f"[INFO] Engine      : {engine_type}")
    print(f"[INFO] Host        : {host}")
    print(f"[INFO] Port        : {port}")
    print(f"[INFO] Database    : {database_name}")
    print(f"[INFO] Username    : {username}")
    print(f"[INFO] SSL Mode    : {ssl_mode}")

    print("[STEP] Building connection URL...")
    url = get_connection_url(
        engine_type=engine_type,
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password_decrypted=password_decrypted,
        ssl_mode=ssl_mode,
    )
    print("[SUCCESS] Connection URL built.")

    print("[STEP] Creating SQLAlchemy engine...")
    engine = create_engine(url)
    print("[SUCCESS] Engine created.")

    try:
        print("[STEP] Creating inspector...")
        inspector = inspect(engine)
        print("[SUCCESS] Inspector created.")

        # Query custom enum types and allowed values directly from database catalogs
        enums_lookup = {}
        if engine_type == DatabaseEngine.postgresql:
            try:
                pg_query = """
                SELECT 
                    ns.nspname AS schema_name,
                    t.relname AS table_name,
                    a.attname AS column_name,
                    e.enumlabel AS enum_value
                FROM pg_attribute a
                JOIN pg_class t ON a.attrelid = t.oid
                JOIN pg_namespace ns ON t.relnamespace = ns.oid
                JOIN pg_type tp ON a.atttypid = tp.oid
                JOIN pg_enum e ON tp.oid = e.enumtypid
                WHERE a.attnum > 0 AND NOT a.attisdropped
                ORDER BY ns.nspname, t.relname, a.attname, e.enumsortorder;
                """
                with engine.connect() as conn:
                    pg_res = conn.execute(text(pg_query)).fetchall()
                    for s_name, t_name, c_name, e_val in pg_res:
                        key = (
                            s_name.lower() if s_name else None,
                            t_name.lower(),
                            c_name.lower(),
                        )
                        if key not in enums_lookup:
                            enums_lookup[key] = []
                        enums_lookup[key].append(e_val)
            except Exception as enum_err:
                print(
                    f"[WARNING] Failed to fetch Postgres enums from catalog: {enum_err}"
                )
            print(f"[INFO] Found {len(enums_lookup)} enum columns in Postgres.")
            print(f"[INFO] Enum lookup keys: {list(enums_lookup.keys())[:5]} ...")

        elif engine_type == DatabaseEngine.mysql:
            try:
                mysql_query = """
                SELECT 
                    table_schema AS schema_name,
                    table_name AS table_name,
                    column_name AS column_name,
                    column_type AS column_type
                FROM information_schema.columns
                WHERE data_type = 'enum' AND table_schema = DATABASE();
                """
                import re

                with engine.connect() as conn:
                    mysql_res = conn.execute(text(mysql_query)).fetchall()
                    for s_name, t_name, c_name, col_type in mysql_res:
                        vals = re.findall(r"'([^']*)'", col_type)
                        if vals:
                            key = (
                                s_name.lower() if s_name else None,
                                t_name.lower(),
                                c_name.lower(),
                            )
                            enums_lookup[key] = vals
            except Exception as enum_err:
                print(
                    f"[WARNING] Failed to fetch MySQL enums from information_schema: {enum_err}"
                )

        schema_data = {"tables": []}

        print("[STEP] Detecting default schema...")
        default_schema = inspector.default_schema_name
        schemas = [default_schema] if default_schema else [None]
        print(f"[INFO] Schemas to inspect: {schemas}")

        for schema in schemas:
            print(f"\n{'-' * 60}")
            print(f"[STEP] Inspecting schema: {schema}")

            try:
                table_names = inspector.get_table_names(schema=schema)
                print(f"[SUCCESS] Found {len(table_names)} tables.")
                print(f"[TABLES] {table_names}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch tables for schema '{schema}'")
                print(e)
                raise

            for table_name in table_names:
                print(f"\n[STEP] Processing table: {table_name}")

                # Columns
                try:
                    print("  [STEP] Fetching columns...")
                    columns = inspector.get_columns(table_name, schema=schema)
                    print(f"  [SUCCESS] Retrieved {len(columns)} columns.")
                except Exception as e:
                    print(f"  [ERROR] Failed fetching columns for table '{table_name}'")
                    print(e)
                    raise

                columns_info = []
                for col in columns:
                    print(f"    [COLUMN] {col['name']} ({col['type']})")
                    col_item = {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                    }
                    lookup_schema = schema.lower() if schema else None
                    key1 = (lookup_schema, table_name.lower(), col["name"].lower())
                    key2 = (None, table_name.lower(), col["name"].lower())
                    allowed_vals = enums_lookup.get(key1) or enums_lookup.get(key2)
                    if allowed_vals:
                        col_item["allowed_values"] = allowed_vals
                    columns_info.append(col_item)

                # Primary Key
                try:
                    print("  [STEP] Fetching primary key...")

                    pk_constraint = inspector.get_pk_constraint(
                        table_name, schema=schema
                    )

                    pk = pk_constraint.get("constrained_columns", [])

                    print(f"  [SUCCESS] Primary Key: {pk}")

                except Exception as e:
                    print(f"  [ERROR] Failed fetching primary key for '{table_name}'")
                    print(e)
                    raise

                # Foreign Keys
                try:
                    print("  [STEP] Fetching foreign keys...")
                    fks = inspector.get_foreign_keys(table_name, schema=schema)
                    print(f"  [SUCCESS] Found {len(fks)} foreign keys.")
                except Exception as e:
                    print(f"  [ERROR] Failed fetching foreign keys for '{table_name}'")
                    print(e)
                    raise

                fks_info = []
                for fk in fks:
                    print(
                        f"    [FK] {fk['constrained_columns']} -> "
                        f"{fk['referred_table']}.{fk['referred_columns']}"
                    )
                    fks_info.append(
                        {
                            "constrained_columns": fk["constrained_columns"],
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"],
                        }
                    )

                schema_data["tables"].append(
                    {
                        "schema": schema,
                        "name": table_name,
                        "columns": columns_info,
                        "primary_key": pk,
                        "foreign_keys": fks_info,
                    }
                )

                print(f"[SUCCESS] Finished processing table: {table_name}")

        print("=" * 80)
        print("[DONE] Schema introspection complete.")
        print(f"[INFO] Total tables processed: {len(schema_data['tables'])}")

        return schema_data

    except Exception as e:
        print("=" * 80)
        print("[FATAL] Schema introspection failed!")
        print(type(e).__name__)
        print(str(e))
        raise

    finally:
        print("[STEP] Disposing SQLAlchemy engine...")
        engine.dispose()
        print("[DONE] Engine disposed.")


# --- Access Policy Utilities ---


def check_user_db_access(db: Session, user: User, connection_id: uuid.UUID) -> bool:
    """
    Returns True if user has access to at least some parts of the database.
    """
    if user.role.is_admin:
        return True

    policy_count = (
        db.query(DatabaseAccessPolicy)
        .filter(
            DatabaseAccessPolicy.connection_id == connection_id,
            DatabaseAccessPolicy.role_id == user.role_id,
        )
        .count()
    )
    return policy_count > 0


def get_user_authorized_tables(
    db: Session, user: User, connection_id: uuid.UUID, all_tables: List[str]
) -> List[str]:
    """
    Returns a list of table names the user is authorized to query.
    If the user has database-level access (table_name IS NULL), they can access all tables.
    Otherwise, access is unioned across all matching policies.
    """
    if user.role.is_admin:
        return all_tables

    policies = (
        db.query(DatabaseAccessPolicy)
        .filter(
            DatabaseAccessPolicy.connection_id == connection_id,
            DatabaseAccessPolicy.role_id == user.role_id,
        )
        .all()
    )

    authorized_tables = set()
    for policy in policies:
        if policy.table_name is None:
            # Grant for entire database
            return all_tables
        authorized_tables.add(policy.table_name)

    return list(authorized_tables)


def _recalculate_policy_inheritance(
    db: Session,
    connection_id: uuid.UUID,
    role_id: uuid.UUID,
    table_name: Optional[str] = None,
) -> None:
    """
    Recalculates access policies dynamically. When direct/dept grants are assigned,
    we create explicit rows for ancestor roles or department members.
    Similar to `_create_policies_with_inheritance` in documents.
    """
    pass  # Managed explicitly in API endpoints to stay simple and identical to documents access creation.


# --- Query Generator & Execution Layer ---


def format_query_results_for_prompt(
    results: Optional[List[Dict[str, Any]]], threshold: int = 20
) -> str:
    """
    Formats SQL query results for inclusion in prompt. If the results list is empty
    or None, returns a message. If the size of the result set exceeds `threshold`,
    returns a bounded summary (row count + first 3 rows).
    """
    import json

    if not results:
        return "No results returned."
    if not isinstance(results, list):
        return str(results)

    total_rows = len(results)
    if total_rows <= threshold:
        return json.dumps(results, indent=2, default=str)

    # Bounded summary: row count + first 3 rows
    summary_rows = results[:3]
    return (
        f"Total rows: {total_rows} (showing first 3 rows as summary)\n"
        f"{json.dumps(summary_rows, indent=2, default=str)}"
    )


def run_query_on_connection(
    connection: ExternalDatabaseConnection,
    sql_query: str,
) -> List[Dict[str, Any]]:
    # Decrypt password and build connection URL
    password_decrypted = decrypt_password(connection.password)
    url = get_connection_url(
        engine_type=connection.engine,
        host=connection.host,
        port=connection.port,
        database_name=connection.database_name,
        username=connection.username,
        password_decrypted=password_decrypted,
        ssl_mode=connection.ssl_mode,
    )

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            # Enforce read-only transaction for PostgreSQL
            if connection.engine == DatabaseEngine.postgresql:
                conn.execute(text("SET TRANSACTION READ ONLY"))

            # Prevent modification queries
            sql_upper = sql_query.upper().strip()
            if not (sql_upper.startswith(("SELECT", "WITH", "EXPLAIN"))):
                raise ValueError("Only read-only SELECT queries are allowed.")

            try:
                result = conn.execute(text(sql_query))

                # Fetch maximum of 100 rows to enforce safety bounds
                rows = result.fetchmany(100)

                # Build output as list of dicts
                output = []
                if result.returns_rows:
                    keys = list(result.keys())
                    for row in rows:
                        output.append(dict(zip(keys, row)))
                return output

            except Exception as e:
                # Detect schema drift and return actionable error
                err_str = str(e).lower()
                is_drift = (
                    ("relation" in err_str and "does not exist" in err_str)
                    or ("column" in err_str and "does not exist" in err_str)
                    or ("table" in err_str and "exist" in err_str)
                    or ("column" in err_str and "field list" in err_str)
                    or ("unknown column" in err_str)
                )
                if is_drift:
                    raise ValueError(
                        f"Database execution failed due to a detected schema mismatch (drift): {str(e)}. "
                        "Please request an Administrator to 'Refresh Schema' for this database."
                    )
                raise e
    finally:
        engine.dispose()


def get_user_authorized_columns_for_table(
    policies: List[DatabaseAccessPolicy], table_name: str, all_table_columns: List[str]
) -> Set[str]:
    """
    Returns a set of lowercase column names the user is authorized to access on a specific table.
    """
    authorized_columns = set()
    has_full_table_access = False
    has_any_matching_policy = False

    for policy in policies:
        if policy.table_name is None or policy.table_name.lower() == table_name.lower():
            has_any_matching_policy = True
            if not policy.columns:  # None or empty means all columns
                has_full_table_access = True
            else:
                for col_spec in policy.columns:
                    if "." in col_spec:
                        tbl, col = col_spec.split(".", 1)
                        if tbl.lower() == table_name.lower():
                            authorized_columns.add(col.lower())
                    else:
                        authorized_columns.add(col_spec.lower())

    if not has_any_matching_policy:
        return set()  # No access to this table

    if has_full_table_access:
        return {c.lower() for c in all_table_columns}

    return authorized_columns


def get_user_filtered_schema(
    db: Session, user: User, connection_id: uuid.UUID, raw_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Filters a raw schema JSON structure based on the user's permissions and policies.
    """
    raw_tables = raw_schema.get("tables", []) if raw_schema else []
    if user.role.is_admin:
        return {"tables": raw_tables}

    all_tables = [t["name"] for t in raw_tables if "name" in t]
    authorized_table_names = get_user_authorized_tables(
        db, user, connection_id, all_tables
    )

    if not authorized_table_names:
        return {"tables": []}

    policies = (
        db.query(DatabaseAccessPolicy)
        .filter(
            DatabaseAccessPolicy.connection_id == connection_id,
            DatabaseAccessPolicy.role_id == user.role_id,
        )
        .all()
    )

    authorized_tables_info = []
    for t in raw_tables:
        t_name = t.get("name")
        if t_name in authorized_table_names:
            all_cols = [c["name"] for c in t.get("columns", [])]
            auth_cols = get_user_authorized_columns_for_table(
                policies, t_name, all_cols
            )

            if auth_cols:
                tbl_copy = dict(t)
                tbl_copy["columns"] = [
                    col
                    for col in t.get("columns", [])
                    if col["name"].lower() in auth_cols
                ]
                authorized_tables_info.append(tbl_copy)

    return {"tables": authorized_tables_info}


def check_sql_authorized_columns(
    sql_query: str,
    engine_type: str,
    authorized_cols_by_table: Dict[str, Set[str]],
    valid_tables: Set[str],
    all_physical_cols_by_table: Optional[Dict[str, Set[str]]] = None,
) -> None:
    """
    Parses a SQL query using sqlglot and checks if it accesses any unauthorized columns or tables.
    Raises ValueError if unauthorized access is detected.
    """
    import sqlglot
    from sqlglot.expressions import Column, Table, CTE

    # Decide SQL dialect
    read_dialect = "postgres" if engine_type.lower() == "postgresql" else "mysql"

    # Parse the SQL query into an AST
    try:
        expression = sqlglot.parse_one(sql_query, read=read_dialect)
    except Exception as e:
        raise ValueError(
            f"SQL validation error: Failed to parse generated SQL query: {e}"
        )

    # Store a mapping of table aliases to their actual table names
    alias_map = {}
    for table_expression in expression.find_all(Table):
        table_name = table_expression.name.lower().strip('"`')
        alias_map[table_name] = table_name
        alias_name = table_expression.alias.lower().strip('"`')
        if alias_name:
            alias_map[alias_name] = table_name

    #  Find and store CTEs
    ctes = {cte.alias.lower().strip('"`') for cte in expression.find_all(CTE)}
    for table_expression in expression.find_all(Table):
        table_name = table_expression.name.lower().strip('"`')
        #  Skip CTEs
        if table_name in ctes:
            continue
        # If table unauthorized, raise an error
        if table_name not in authorized_cols_by_table:
            raise ValueError(f"Access denied: Table '{table_name}' is unauthorized.")

    # Iterate through every column referenced anywhere in the SQL query and validate its access permissions
    for col_expression in expression.find_all(Column):
        col_name = col_expression.name.lower().strip('"`')
        table_alias = col_expression.text("table").lower().strip('"`')

        resolved_table = None
        if table_alias:
            resolved_table = alias_map.get(table_alias)
            if not resolved_table:
                resolved_table = table_alias
        else:
            matching_tables = []
            for t_name in alias_map.items():
                if (
                    t_name in authorized_cols_by_table
                    and col_name in authorized_cols_by_table[t_name]
                ):
                    matching_tables.append(t_name)

            if len(matching_tables) == 1:
                resolved_table = matching_tables[0]
            elif len(matching_tables) > 1:
                for mt in matching_tables:
                    if col_name not in authorized_cols_by_table[mt]:
                        raise ValueError(
                            f"Access denied: Column '{col_name}' on table '{mt}' is unauthorized."
                        )
                continue
            else:
                for t_name in alias_map.items():
                    if t_name in valid_tables:
                        resolved_table = t_name
                        break

        if resolved_table:
            resolved_table = resolved_table.strip('"`')
            # If we know the physical columns of the table, only check if col_name is actually one of them
            if (
                all_physical_cols_by_table
                and resolved_table in all_physical_cols_by_table
            ):
                if col_name not in all_physical_cols_by_table[resolved_table]:
                    # This is an alias, CTE, function, or non-physical column. Skip validation.
                    continue

            if resolved_table in authorized_cols_by_table:
                allowed_cols = authorized_cols_by_table[resolved_table]
                if col_name not in allowed_cols:
                    raise ValueError(
                        f"Access denied: Column '{col_name}' on table '{resolved_table}' is unauthorized."
                    )
            elif resolved_table in valid_tables:
                raise ValueError(
                    f"Access denied: Table '{resolved_table}' is unauthorized."
                )
