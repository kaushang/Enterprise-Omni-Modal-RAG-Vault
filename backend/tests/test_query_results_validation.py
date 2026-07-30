import uuid
from datetime import datetime, timezone
from app.schemas.chat import MessageResponse


def test_message_response_with_string_query_results():
    """Verify that MessageResponse handles string values like 'No rows found' in query_results without error."""
    msg_data = {
        "id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "role": "assistant",
        "content": "Here are the query results.",
        "created_at": datetime.now(timezone.utc),
        "citations": [],
        "query_results": "No rows found",
    }

    response = MessageResponse.model_validate(msg_data)
    assert response.query_results is None


def test_message_response_with_json_string_query_results():
    """Verify that MessageResponse parses valid JSON string query_results."""
    msg_data = {
        "id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "role": "assistant",
        "content": "Here are the query results.",
        "created_at": datetime.now(timezone.utc),
        "citations": [],
        "query_results": '[{"col1": "val1"}]',
    }

    response = MessageResponse.model_validate(msg_data)
    assert response.query_results == [{"col1": "val1"}]


def test_message_response_with_list_query_results():
    """Verify that MessageResponse accepts native list of dict query_results."""
    rows = [{"id": 1, "name": "Alice"}]
    msg_data = {
        "id": uuid.uuid4(),
        "session_id": uuid.uuid4(),
        "role": "assistant",
        "content": "Here are the query results.",
        "created_at": datetime.now(timezone.utc),
        "citations": [],
        "query_results": rows,
    }

    response = MessageResponse.model_validate(msg_data)
    assert response.query_results == rows
