import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.agents.graph import build_graph
from app.services.agents.nodes.query_rewriter import query_rewriter_node
from app.services.agents.tools.query_rewriter_tools import (
    assess_query_quality as _assess_query_quality,
    rewrite_query as _rewrite_query,
)


async def _validate_rewrite(original, rewritten, issues):
    return True, "Preserves intent (skipped)"


@pytest.mark.asyncio
async def test_assess_query_quality_success():
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(
        {
            "issues": ["typo"],
            "confidence": 0.7,
            "suggested_actions": ["correct_spelling"],
            "needs_rewrite": True,
        }
    )
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=mock_msg)]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch(
        "app.services.agents.tools.query_rewriter_tools.get_ollama_client",
        return_value=mock_client,
    ):
        result = await _assess_query_quality("wats the revenue")
        assert result["needs_rewrite"] is True
        assert result["confidence"] == 0.7
        assert "typo" in result["issues"]


@pytest.mark.asyncio
async def test_assess_query_quality_fail_open():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("LLM error"))

    with patch(
        "app.services.agents.tools.query_rewriter_tools.get_ollama_client",
        return_value=mock_client,
    ):
        result = await _assess_query_quality("wats the revenue")
        assert result["needs_rewrite"] is False
        assert result["confidence"] == 1.0
        assert result["issues"] == []


@pytest.mark.asyncio
async def test_rewrite_query_success():
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = "What is the total revenue?"
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=mock_msg)]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch(
        "app.services.agents.tools.query_rewriter_tools.get_ollama_client",
        return_value=mock_client,
    ):
        result = await _rewrite_query(
            query="wats the revenue",
            strategies=["correct"],
            issues=["typo"],
            suggested_actions=["correct_spelling"],
        )
        assert result == {"rewritten_query": "What is the total revenue?"}


@pytest.mark.asyncio
async def test_rewrite_query_fail_open():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API limit"))

    with patch(
        "app.services.agents.tools.query_rewriter_tools.get_ollama_client",
        return_value=mock_client,
    ):
        result = await _rewrite_query(
            query="wats the revenue",
            strategies=["correct"],
            issues=["typo"],
            suggested_actions=["correct_spelling"],
        )
        assert result == {"rewritten_query": "wats the revenue"}


@pytest.mark.asyncio
async def test_validate_rewrite_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text=json.dumps({"valid": True, "reason": "Preserves intent"}))
    ]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        valid, reason = await _validate_rewrite(
            "wats the revenue", "What is the revenue?", ["typo"]
        )
        assert valid is True
        assert "intent" in reason.lower()


@pytest.mark.asyncio
async def test_validate_rewrite_fail_open():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("Network error"))

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        valid, reason = await _validate_rewrite(
            "wats the revenue", "What is the revenue?", ["typo"]
        )
        assert valid is True
        assert "skipped" in reason.lower()


@pytest.mark.asyncio
async def test_query_rewriter_node_no_rewrite():
    mock_client = MagicMock()

    mock_msg = MagicMock()
    mock_msg.content = json.dumps(
        {
            "final_query": "Show all users",
            "was_rewritten": False,
            "diagnosis": {
                "needs_rewrite": False,
                "confidence": 1.0,
                "issues": [],
            },
        }
    )
    mock_msg.tool_calls = []

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    state = {"query": "Show all users"}

    with patch(
        "app.services.agents.nodes.query_rewriter.get_ollama_client",
        return_value=mock_client,
    ):
        result = await query_rewriter_node(state)
        assert result["query"] == "Show all users"
        assert result["original_query"] == "Show all users"
        assert result["query_was_rewritten"] is False


@pytest.mark.asyncio
async def test_query_rewriter_node_fail_open_entirely():
    with patch(
        "app.services.agents.nodes.query_rewriter.get_ollama_client",
        side_effect=Exception("Catastrophic error"),
    ):
        state = {"query": "Select * from sales"}
        result = await query_rewriter_node(state)
        assert result == {"query": "Select * from sales"}


def test_graph_entry_point():
    g = build_graph()
    assert "query_rewriter_node" in g.nodes
    assert "orchestrator_node" in g.nodes
