import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.services.agents.graph import build_graph
from app.services.agents.nodes.query_rewriter import (
    _assess_query_quality,
    _rewrite_query,
    _validate_rewrite,
    query_rewriter_node,
)


@pytest.mark.asyncio
async def test_assess_query_quality_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text=json.dumps(
                {
                    "issues": ["typo"],
                    "confidence": 0.7,
                    "suggested_actions": ["correct_spelling"],
                    "needs_rewrite": True,
                }
            )
        )
    ]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        result = await _assess_query_quality("wats the revenue")
        assert result["needs_rewrite"] is True
        assert result["confidence"] == 0.7
        assert "typo" in result["issues"]


@pytest.mark.asyncio
async def test_assess_query_quality_fail_open():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("LLM error"))

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        result = await _assess_query_quality("wats the revenue")
        assert result["needs_rewrite"] is False
        assert result["confidence"] == 1.0
        assert result["issues"] == []


@pytest.mark.asyncio
async def test_rewrite_query_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="What is the total revenue?")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        result = await _rewrite_query(
            query="wats the revenue",
            strategy="correct",
            issues=["typo"],
            suggested_actions=["correct_spelling"],
        )
        assert result == "What is the total revenue?"


@pytest.mark.asyncio
async def test_rewrite_query_fail_open():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("API limit"))

    with patch(
        "app.services.rag_service._get_async_anthropic_client", return_value=mock_client
    ):
        result = await _rewrite_query(
            query="wats the revenue",
            strategy="correct",
            issues=["typo"],
            suggested_actions=["correct_spelling"],
        )
        assert result == "wats the revenue"


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
        assert reason == "Preserves intent"


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

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "assess_query_quality"
    tool_block.id = "call_1"
    tool_block.input = {"query": "Show all users"}

    mock_resp1 = MagicMock()
    mock_resp1.content = [tool_block]

    mock_resp2 = MagicMock()
    mock_resp2.content = [
        MagicMock(
            text=json.dumps(
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
        )
    ]

    mock_client.messages.create = AsyncMock(side_effect=[mock_resp1, mock_resp2])

    state = {"query": "Show all users"}

    with (
        patch(
            "app.services.rag_service._get_async_anthropic_client",
            return_value=mock_client,
        ),
        patch(
            "app.services.agents.nodes.query_rewriter._assess_query_quality",
            new=AsyncMock(
                return_value={
                    "issues": [],
                    "confidence": 1.0,
                    "suggested_actions": [],
                    "needs_rewrite": False,
                }
            ),
        ),
    ):
        result = await query_rewriter_node(state)
        assert result["query"] == "Show all users"
        assert result["original_query"] == "Show all users"
        assert result["query_was_rewritten"] is False


@pytest.mark.asyncio
async def test_query_rewriter_node_fail_open_entirely():
    with patch(
        "app.services.rag_service._get_async_anthropic_client",
        side_effect=Exception("Catastrophic error"),
    ):
        state = {"query": "Select * from sales"}
        result = await query_rewriter_node(state)
        assert result == {"query": "Select * from sales"}


def test_graph_entry_point():
    g = build_graph()
    assert "query_rewriter_node" in g.nodes
    assert "orchestrator_node" in g.nodes
