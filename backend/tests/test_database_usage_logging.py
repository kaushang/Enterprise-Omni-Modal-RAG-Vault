import sys
import os
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base
from app.models.tenant import Tenant
from app.models.role import Role
from app.models.user import User
from app.models.external_database import ExternalDatabaseConnection, DatabaseSchemaCache
from app.models.usage_log import UsageLog
from app.services.rag_service import run_rag_pipeline

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


@pytest.mark.asyncio
@patch("app.services.rag_service._async_anthropic_client")
@patch("app.services.database_service._async_anthropic_client")
@patch("app.services.database_service.check_user_db_access")
@patch("app.services.database_service.get_user_authorized_tables")
@patch("app.services.database_service.run_query_on_connection")
async def test_run_rag_pipeline_database_logs_usage(
    mock_run_query,
    mock_get_tables,
    mock_db_access,
    mock_db_anthropic_client,
    mock_rag_anthropic_client,
    db,
):
    # Setup mock user, role, tenant, database connection and schema cache
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    db.add(tenant)
    db.commit()

    role = Role(id=uuid.uuid4(), tenant_id=tenant.id, name="Admin", is_admin=True)
    db.add(role)
    db.commit()

    user = User(
        id=uuid.uuid4(),
        email="user@example.com",
        full_name="Test User",
        hashed_password="...",
        tenant_id=tenant.id,
        role_id=role.id,
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
        username="postgres",
        password="encrypted_password",
        database_name="test_db",
    )
    db.add(connection)
    db.commit()

    schema_cache = DatabaseSchemaCache(
        connection_id=connection.id,
        schema_data={
            "tables": [
                {
                    "name": "roles",
                    "columns": [{"name": "id", "type": "UUID"}],
                    "primary_key": [],
                    "foreign_keys": [],
                }
            ]
        },
    )
    db.add(schema_cache)
    db.commit()

    # Set up mocks
    mock_db_access.return_value = True
    mock_get_tables.return_value = ["roles"]
    mock_run_query.return_value = [{"count": 5}]

    # Mock Translation LLM Response
    mock_translate_resp = MagicMock()
    mock_translate_resp.content = [MagicMock(text="SELECT * FROM roles;")]
    mock_translate_resp.usage = MagicMock(input_tokens=150, output_tokens=75)
    mock_db_anthropic_client.messages.create = AsyncMock(
        return_value=mock_translate_resp
    )

    # Mock Summarization stream/message
    mock_final_msg = MagicMock()
    mock_final_msg.usage = MagicMock(input_tokens=200, output_tokens=120)

    class MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def __aiter__(self):
            # yield text event
            event = MagicMock()
            event.type = "text"
            event.text = "There are 5 roles."
            yield event

        async def get_final_message(self):
            return mock_final_msg

    mock_rag_anthropic_client.messages.stream.return_value = MockStream()

    # Execute RAG pipeline
    events = []
    async for event in run_rag_pipeline(
        query="How many roles?",
        user=user,
        db=db,
        database_id=connection.id,
    ):
        events.append(event)

    # Verify RAG response events
    done_event = [e for e in events if e.get("type") == "done"]
    assert len(done_event) == 1
    assert done_event[0]["answer"] == "There are 5 roles."

    # Verify BOTH UsageLog records exist (one for translation, one for summarization)
    logs = db.query(UsageLog).filter(UsageLog.user_id == user.id).all()
    assert len(logs) == 2

    # Check first log (SQL Translation)
    translation_log = [l for l in logs if l.input_tokens == 150][0]  # noqa: E741
    assert translation_log.output_tokens == 75
    assert translation_log.provider == "anthropic"

    # Check second log (SQL Summarization)
    summarization_log = [l for l in logs if l.input_tokens == 200][0]  # noqa: E741
    assert summarization_log.output_tokens == 120
    assert summarization_log.provider == "anthropic"
