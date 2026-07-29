import sys
import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base
from app.models.role import Role
from app.models.user import User
from app.models.tenant import Tenant
from app.models.query_session import QuerySession
from app.models.query_message import QueryMessage
from app.models.query_citation import QueryCitation
from app.models.document import Document
from app.models.external_database import ExternalDatabaseConnection, DatabaseSchemaCache
from app.schemas.chat import QueryRequest
from app.api.v1.chat import send_query
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

    # Create default role, tenant, user
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    session.add(tenant)

    role = Role(id=uuid.uuid4(), name="member", is_admin=True, tenant_id=tenant.id)
    session.add(role)
    session.commit()

    user = User(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        email="test@example.com",
        full_name="Test User",
        hashed_password="hash",
        role_id=role.id,
        is_active=True,
    )
    session.add(user)
    session.commit()

    try:
        yield (session, user, tenant)
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.mark.asyncio
async def test_database_to_document_fallback_translation_failure(db):
    session, user, tenant = db

    # 1. Create a query session
    query_session = QuerySession(
        user_id=user.id, tenant_id=tenant.id, title="Fallback Chat"
    )
    session.add(query_session)

    # Create external database connection
    db_conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        name="Test DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="test_db",
        username="postgres",
        password="pwd",
        status="active",
    )
    session.add(db_conn)

    # Cache database schema (with tables)
    schema_cache = DatabaseSchemaCache(
        connection_id=db_conn.id,
        schema_data={
            "tables": [
                {
                    "name": "users",
                    "columns": [{"name": "id", "type": "INTEGER"}],
                    "primary_key": ["id"],
                    "foreign_keys": [],
                }
            ]
        },
    )
    session.add(schema_cache)
    session.commit()

    # Create document
    doc = Document(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        uploaded_by=user.id,
        filename="fallback_doc.pdf",
        file_type="pdf",
        status="ready",
        owner_type="organisation",
        visibility="public",
        chunk_count=1,
        qdrant_collection="tenant_test",
    )
    session.add(doc)
    session.commit()

    # Mock vectors search, embedding, and LLM call
    async def mock_generator(*args, **kwargs):
        yield "token", "Response from fallback documents."

    mock_embed = lambda t: [0.1] * 1024  # noqa: E731
    mock_search = lambda *args, **kwargs: [  # noqa: E731
        {
            "id": "vec-1",
            "score": 0.9,
            "payload": {
                "document_id": str(doc.id),
                "chunk_text": "Company remote work policy details",
                "chunk_index": 0,
            },
        }
    ]

    with (
        patch(
            "app.services.agents.tools.sql_tools.generate_sql",
            side_effect=ValueError("Ambiguous column"),
        ),
        patch("app.services.embedding_service.embed_text", side_effect=mock_embed),
        patch("app.services.rag_service.search_vectors", side_effect=mock_search),
        patch(
            "app.services.rag_service._execute_llm_stream", side_effect=mock_generator
        ),
        patch("app.services.rag_service._get_cross_encoder", return_value=None),
        patch(
            "app.services.agents.nodes.answer_generation_node.get_db_session",
            return_value=session,
        ),
    ):
        events = []
        async for event in run_rag_pipeline(
            query="Find employee salaries",
            user=user,
            db=session,
            database_id=db_conn.id,
            session_id=query_session.id,
        ):
            events.append(event)

        # The first events should be fallback warning tokens
        fallback_token_events = [e["content"] for e in events if e["type"] == "token"]
        assert any(
            "Could not translate query to SQL. Searching documents..." in content
            for content in fallback_token_events
        )

        # The last event should be a done event with citations pointing to fallback_doc.pdf
        done_event = events[-1]
        assert done_event["type"] == "done"
        assert len(done_event["citations"]) == 1
        assert done_event["citations"][0]["filename"] == "fallback_doc.pdf"


@pytest.mark.asyncio
async def test_database_to_document_fallback_empty_results(db):
    session, user, tenant = db

    # 1. Create a query session
    query_session = QuerySession(
        user_id=user.id, tenant_id=tenant.id, title="Fallback Chat Empty"
    )
    session.add(query_session)

    # Create external database connection
    db_conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        name="Test DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="test_db",
        username="postgres",
        password="pwd",
        status="active",
    )
    session.add(db_conn)

    # Cache database schema
    schema_cache = DatabaseSchemaCache(
        connection_id=db_conn.id,
        schema_data={
            "tables": [
                {
                    "name": "users",
                    "columns": [{"name": "id", "type": "INTEGER"}],
                    "primary_key": ["id"],
                    "foreign_keys": [],
                }
            ]
        },
    )
    session.add(schema_cache)
    session.commit()

    # Create document
    doc = Document(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        uploaded_by=user.id,
        filename="fallback_doc_2.pdf",
        file_type="pdf",
        status="ready",
        owner_type="organisation",
        visibility="public",
        chunk_count=1,
        qdrant_collection="tenant_test",
    )
    session.add(doc)
    session.commit()

    # Mock SQL execution returning empty results
    async def mock_generator(*args, **kwargs):
        yield "token", "Response from fallback documents empty check."

    mock_embed = lambda t: [0.1] * 1024  # noqa: E731
    mock_search = lambda *args, **kwargs: [  # noqa: E731
        {
            "id": "vec-1",
            "score": 0.9,
            "payload": {
                "document_id": str(doc.id),
                "chunk_text": "Empty database fallback details",
                "chunk_index": 0,
            },
        }
    ]

    with (
        patch(
            "app.services.agents.tools.sql_tools.generate_sql",
            return_value="SELECT * FROM users;",
        ),
        patch(
            "app.services.database_service.run_query_on_connection", return_value=[]
        ),  # returns empty results
        patch("app.services.embedding_service.embed_text", side_effect=mock_embed),
        patch("app.services.rag_service.search_vectors", side_effect=mock_search),
        patch(
            "app.services.rag_service._execute_llm_stream", side_effect=mock_generator
        ),
        patch("app.services.rag_service._get_cross_encoder", return_value=None),
        patch(
            "app.services.agents.nodes.answer_generation_node.get_db_session",
            return_value=session,
        ),
    ):
        events = []
        async for event in run_rag_pipeline(
            query="Find empty users",
            user=user,
            db=session,
            database_id=db_conn.id,
            session_id=query_session.id,
        ):
            events.append(event)

        # The first events should be fallback warning tokens
        fallback_token_events = [e["content"] for e in events if e["type"] == "token"]
        assert any(
            "No matching records found in database. Searching documents..." in content
            for content in fallback_token_events
        )

        # The last event should be a done event with citations pointing to fallback_doc_2.pdf
        done_event = events[-1]
        assert done_event["type"] == "done"
        assert len(done_event["citations"]) == 1
        assert done_event["citations"][0]["filename"] == "fallback_doc_2.pdf"


@pytest.mark.asyncio
async def test_multi_turn_history_formatting(db):
    session, user, tenant = db

    # 1. Create a query session
    query_session = QuerySession(
        user_id=user.id, tenant_id=tenant.id, title="Multi Turn Chat"
    )
    session.add(query_session)
    session.commit()

    # 2. Add an assistant message with SQL query and results
    db_msg = QueryMessage(
        session_id=query_session.id,
        role="assistant",
        content="Here is the user count from database.",
        generated_sql="SELECT count(*) FROM users;",
        query_results=[{"count": 5}],
    )
    session.add(db_msg)
    session.commit()

    # 3. Add a document
    doc = Document(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        uploaded_by=user.id,
        filename="policy.pdf",
        file_type="pdf",
        status="ready",
        owner_type="organisation",
        visibility="public",
        chunk_count=1,
        qdrant_collection="tenant_test",
    )
    session.add(doc)
    session.commit()

    # 4. Add an assistant message with document citations
    doc_msg = QueryMessage(
        session_id=query_session.id,
        role="assistant",
        content="Our remote work policy permits 3 days work from home.",
    )
    session.add(doc_msg)
    session.commit()

    doc_citation = QueryCitation(
        message_id=doc_msg.id,
        document_id=doc.id,
        qdrant_vector_id="vec-1",
        chunk_text="Employees may work remotely up to three days per week.",
        chunk_index=0,
    )
    session.add(doc_citation)
    session.commit()

    # 5. Mock run_rag_pipeline to check the conversation history format
    mock_run_rag = MagicMock()

    async def mock_generator(*args, **kwargs):
        yield {"type": "token", "content": "Analyzing..."}
        yield {
            "type": "done",
            "answer": "Answer based on context.",
            "citations": [],
        }

    mock_run_rag.side_effect = mock_generator

    body = QueryRequest(content="Can I work from home 4 days?")

    with (
        patch("app.api.v1.chat.run_rag_pipeline", new=mock_run_rag),
        patch(
            "app.services.rate_limit_service.check_and_increment_rate_limit",
            return_value=True,
        ),
    ):
        response = await send_query(
            session_id=query_session.id,
            body=body,
            current_user=user,
            db=session,
        )
        async for chunk in response.body_iterator:
            pass

        # Verify how the conversation history was formatted
        called_kwargs = mock_run_rag.call_args[1]
        history = called_kwargs["conversation_history"]

        # Ensure both document chunks and SQL results are formatted in the multi-turn context!
        assert "Generated SQL:\nSELECT count(*) FROM users;" in history
        assert 'SQL Results:\n[\n  {\n    "count": 5\n  }\n]' in history
        assert (
            "Retrieved Document Chunks:\nSource: policy.pdf | Chunk: 0\nEmployees may work remotely up to three days per week."
            in history
        )
        assert (
            "Answer: Our remote work policy permits 3 days work from home." in history
        )


@pytest.mark.asyncio
async def test_cross_source_fusion_success(db):
    session, user, tenant = db

    # 1. Create a query session
    query_session = QuerySession(
        user_id=user.id, tenant_id=tenant.id, title="Cross Source Fusion Chat"
    )
    session.add(query_session)

    # Create external database connection
    db_conn = ExternalDatabaseConnection(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        name="User DB",
        engine="postgresql",
        host="localhost",
        port=5432,
        database_name="user_db",
        username="postgres",
        password="pwd",
        status="active",
    )
    session.add(db_conn)

    # Cache database schema (with tables)
    schema_cache = DatabaseSchemaCache(
        connection_id=db_conn.id,
        schema_data={
            "tables": [
                {
                    "name": "users",
                    "columns": [{"name": "id", "type": "INTEGER"}],
                    "primary_key": ["id"],
                    "foreign_keys": [],
                }
            ]
        },
    )
    session.add(schema_cache)
    session.commit()

    # Create document
    doc = Document(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        uploaded_by=user.id,
        filename="company_policy.pdf",
        file_type="pdf",
        status="ready",
        owner_type="organisation",
        visibility="public",
        chunk_count=1,
        qdrant_collection="tenant_test",
    )
    session.add(doc)
    session.commit()

    # Mock vectors search, embedding, and LLM call
    async def mock_generator(*args, **kwargs):
        yield "token", "Merged answer from DB and document."
        yield "usage", {"input_tokens": 10, "output_tokens": 20}

    mock_embed = lambda t: [0.1] * 1024  # noqa: E731
    mock_search = lambda *args, **kwargs: [  # noqa: E731
        {
            "id": "vec-1",
            "score": 0.9,
            "payload": {
                "document_id": str(doc.id),
                "chunk_text": "Employees must wear badge.",
                "chunk_index": 0,
            },
        }
    ]

    with (
        patch(
            "app.services.agents.tools.sql_tools.generate_sql",
            return_value="SELECT count(*) FROM users;",
        ),
        patch(
            "app.services.database_service.run_query_on_connection",
            return_value=[{"count": 10}],
        ),
        patch("app.services.embedding_service.embed_text", side_effect=mock_embed),
        patch("app.services.rag_service.search_vectors", side_effect=mock_search),
        patch(
            "app.services.rag_service._execute_llm_stream", side_effect=mock_generator
        ),
        patch("app.services.rag_service._get_cross_encoder", return_value=None),
        patch(
            "app.services.agents.nodes.answer_generation_node.get_db_session",
            return_value=session,
        ),
    ):
        events = []
        async for event in run_rag_pipeline(
            query="compare database users count vs document rules",
            user=user,
            db=session,
            database_id=db_conn.id,
            document_id=doc.id,
            session_id=query_session.id,
        ):
            events.append(event)

        # The last event should be a done event with citations from both sources
        done_event = events[-1]
        assert done_event["type"] == "done"
        assert done_event["status"] == "success"

        # Check citations
        citations = done_event["citations"]
        assert len(citations) == 2

        # DB citation
        db_citation = next((c for c in citations if "Database:" in c["filename"]), None)
        assert db_citation is not None
        assert db_citation["filename"] == "Database: User DB"
        assert "SQL: SELECT count(*)" in db_citation["chunk_text"]

        # Doc citation
        doc_citation = next(
            (c for c in citations if c["filename"] == "company_policy.pdf"), None
        )
        assert doc_citation is not None
        assert doc_citation["chunk_text"] == "Employees must wear badge."
