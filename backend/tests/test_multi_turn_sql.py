import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base
from app.models.user import User
from app.models.tenant import Tenant
from app.models.role import Role
from app.models.query_session import QuerySession
from app.models.query_message import QueryMessage
from app.models.enums import MessageRole
from app.models.external_database import (
    ExternalDatabaseConnection,
    DatabaseSchemaCache,
    DatabaseAccessPolicy,
)
from app.services.rag_service import get_recent_turns
from app.services.agents.tools.sql_tools import generate_sql

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


@pytest.fixture
def mock_setup(db):
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    db.add(tenant)
    db.commit()

    role = Role(id=uuid.uuid4(), name="Admin", tenant_id=tenant.id)
    db.add(role)
    db.commit()

    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
        hashed_password="hash",
        tenant_id=tenant.id,
        role_id=role.id,
        is_active=True,
    )
    db.add(user)
    db.commit()

    connection = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        name="Test DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="test_db",
        username="user",
        password="encrypted_password",
        status="active",
    )
    db.add(connection)
    db.commit()

    # grant access to db
    policy = DatabaseAccessPolicy(
        id=uuid.uuid4(),
        connection_id=connection.id,
        role_id=role.id,
        granted_via="direct",
    )
    db.add(policy)
    db.commit()

    schema_cache = DatabaseSchemaCache(
        connection_id=connection.id,
        schema_data={
            "tables": [
                {
                    "name": "employees",
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "name", "type": "VARCHAR"},
                        {"name": "role", "type": "VARCHAR"},
                        {"name": "manager_id", "type": "INTEGER"},
                    ],
                    "primary_key": ["id"],
                    "foreign_keys": [],
                }
            ]
        },
    )
    db.add(schema_cache)
    db.commit()

    session = QuerySession(
        id=uuid.uuid4(), user_id=user.id, tenant_id=tenant.id, title="Test Session"
    )
    db.add(session)
    db.commit()

    return {
        "tenant": tenant,
        "user": user,
        "connection": connection,
        "session": session,
    }


@pytest.mark.asyncio
async def test_two_turn_conversation_pronoun_resolution(db, mock_setup):
    setup = mock_setup
    session = setup["session"]

    # Turn 1: user asks who uploaded the private file
    # We save this message history
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    user_msg_1 = QueryMessage(
        session_id=session.id,
        role=MessageRole.user,
        content="Who is the manager?",
        created_at=now - timedelta(seconds=10),
    )
    assistant_msg_1 = QueryMessage(
        session_id=session.id,
        role=MessageRole.assistant,
        content="The manager is John Doe.",
        created_at=now - timedelta(seconds=5),
        generated_sql="SELECT name FROM employees WHERE role = 'Manager' LIMIT 100",
        query_results=[{"name": "John Doe", "role": "Manager"}],
    )
    db.add_all([user_msg_1, assistant_msg_1])
    db.commit()

    # Turn 2: user asks a follow-up "Who does he report to?" referencing "he" (John Doe)
    user_msg_2 = QueryMessage(
        session_id=session.id,
        role=MessageRole.user,
        content="Who does he report to?",
        created_at=now,
    )
    db.add(user_msg_2)
    db.commit()

    # Mock Anthropic Async client
    with patch("app.services.database_service.AsyncAnthropic") as mock_anthropic_class:
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        async def mock_create_fn(*args, **kwargs):
            m_res = MagicMock()
            m_res.content = [
                MagicMock(
                    text="SELECT name FROM employees WHERE id = (SELECT manager_id FROM employees WHERE name = 'John Doe') LIMIT 100"
                )
            ]
            return m_res

        mock_client.messages.create = mock_create_fn

        with patch(
            "app.services.database_service._async_anthropic_client", new=mock_client
        ):
            # Run translate_nl_to_sql or the entire pipeline
            turns = get_recent_turns(db, session.id)
            assert len(turns) == 1
            assert turns[0]["question"] == "Who is the manager?"
            assert turns[0]["query_results"] == [
                {"name": "John Doe", "role": "Manager"}
            ]

            schema_data_filtered = {
                "tables": [
                    {
                        "name": "employees",
                        "columns": [
                            {"name": "id", "type": "INTEGER"},
                            {"name": "name", "type": "VARCHAR"},
                            {"name": "role", "type": "VARCHAR"},
                            {"name": "manager_id", "type": "INTEGER"},
                        ],
                        "primary_key": ["id"],
                        "foreign_keys": [],
                    }
                ]
            }

            sql = await generate_sql(
                query="Who does he report to?",
                schema=schema_data_filtered,
                engine_type="postgresql",
                conversation_history=turns,
            )

            assert "SELECT" in sql


@pytest.mark.asyncio
async def test_ambiguity_fallback_with_no_context(db):

    # Question: "Who does he report to?" without prior context
    # Mock LLM to return exactly: "I cannot generate a SQL query, this is ambiguous"
    mock_client = MagicMock()

    async def mock_create_fn(*args, **kwargs):
        m_res = MagicMock()
        m_res.content = [
            MagicMock(text="I cannot generate a SQL query, this is ambiguous")
        ]
        return m_res

    mock_client.messages.create = mock_create_fn

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        schema_data_filtered = {
            "tables": [
                {
                    "name": "employees",
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "name", "type": "VARCHAR"},
                    ],
                    "primary_key": ["id"],
                    "foreign_keys": [],
                }
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            await generate_sql(
                query="Who does he report to?",
                schema=schema_data_filtered,
                engine_type="postgresql",
                conversation_history=[],
            )
        assert "I cannot generate a SQL query, this is ambiguous" in str(exc_info.value)


@pytest.mark.asyncio
async def test_large_result_prior_turn_summary(db, mock_setup):
    setup = mock_setup
    session = setup["session"]

    # Create Turn 1 with 25 rows (exceeds threshold of 20)
    large_results = [{"id": i, "name": f"Employee {i}"} for i in range(1, 26)]
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    user_msg_1 = QueryMessage(
        session_id=session.id,
        role=MessageRole.user,
        content="Show all employees",
        created_at=now - timedelta(seconds=10),
    )
    assistant_msg_1 = QueryMessage(
        session_id=session.id,
        role=MessageRole.assistant,
        content="Here are the employees.",
        created_at=now - timedelta(seconds=5),
        generated_sql="SELECT * FROM employees",
        query_results=large_results,
    )
    db.add_all([user_msg_1, assistant_msg_1])
    db.commit()

    turns = get_recent_turns(db, session.id)
    assert len(turns) == 1

    from app.services.database_service import format_query_results_for_prompt

    formatted_results = format_query_results_for_prompt(
        turns[0]["query_results"], threshold=20
    )

    assert "Total rows: 25 (showing first 3 rows as summary)" in formatted_results
    assert "Employee 10" not in formatted_results
    assert "Employee 1" in formatted_results
