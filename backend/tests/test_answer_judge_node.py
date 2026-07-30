from unittest.mock import AsyncMock, patch

import pytest
from app.services.agents.graph import route_after_answer_judge
from app.services.agents.nodes.answer_judge_node import answer_judge_node
from app.services.agents.types import AnswerJudgeResult


@pytest.mark.asyncio
async def test_answer_judge_empty_final_answer():
    state = {
        "original_query": "What is the capital of France?",
        "final_answer": "",
        "answer_judge_attempts": 0,
    }
    result = await answer_judge_node(state)
    assert result["answer_judge_attempts"] == 1
    assert result["low_confidence"] is False
    assert result["answer_judge_result"].passed is False
    assert result["answer_judge_feedback"] == "The generated answer is empty."


@pytest.mark.asyncio
async def test_answer_judge_passed():
    state = {
        "original_query": "What is the capital of France?",
        "final_answer": "The capital of France is Paris.",
        "answer_judge_attempts": 0,
        "rag_result": "Paris is the capital of France.",
    }

    mock_msg = AsyncMock()
    mock_msg.content = '{"passed": true, "reasoning": "Grounded and complete", "grounding_issues": [], "missing_intents": [], "feedback": null}'
    mock_choice = AsyncMock()
    mock_choice.message = mock_msg
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch(
        "app.services.agents.nodes.answer_judge_node.get_ollama_client",
        return_value=mock_client,
    ):
        result = await answer_judge_node(state)

    assert result["answer_judge_attempts"] == 1
    assert result["low_confidence"] is False
    assert result["answer_judge_result"].passed is True
    assert result["answer_judge_feedback"] is None


@pytest.mark.asyncio
async def test_answer_judge_failed_first_attempt():
    state = {
        "original_query": "What is the capital of France and its population?",
        "final_answer": "The capital of France is Paris.",
        "answer_judge_attempts": 0,
        "rag_result": "Paris is the capital of France.",
    }

    mock_msg = AsyncMock()
    mock_msg.content = '{"passed": false, "reasoning": "Missing population", "grounding_issues": [], "missing_intents": ["population"], "feedback": "Include population info."}'
    mock_choice = AsyncMock()
    mock_choice.message = mock_msg
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch(
        "app.services.agents.nodes.answer_judge_node.get_ollama_client",
        return_value=mock_client,
    ):
        result = await answer_judge_node(state)

    assert result["answer_judge_attempts"] == 1
    assert result["low_confidence"] is False
    assert result["answer_judge_result"].passed is False
    assert result["answer_judge_feedback"] == "Include population info."


@pytest.mark.asyncio
async def test_answer_judge_max_attempts_force_pass():
    state = {
        "original_query": "What is the capital of France and its population?",
        "final_answer": "The capital of France is Paris.",
        "answer_judge_attempts": 1,
        "rag_result": "Paris is the capital of France.",
    }

    mock_msg = AsyncMock()
    mock_msg.content = '{"passed": false, "reasoning": "Missing population", "grounding_issues": [], "missing_intents": ["population"], "feedback": "Include population info."}'
    mock_choice = AsyncMock()
    mock_choice.message = mock_msg
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch(
        "app.services.agents.nodes.answer_judge_node.get_ollama_client",
        return_value=mock_client,
    ):
        result = await answer_judge_node(state)

    assert result["answer_judge_attempts"] == 2
    assert result["low_confidence"] is True
    assert result["answer_judge_result"].passed is True
    assert result["answer_judge_feedback"] is None


@pytest.mark.asyncio
async def test_answer_judge_error_soft_fallback():
    state = {
        "original_query": "Test query",
        "final_answer": "Test answer",
        "answer_judge_attempts": 0,
    }

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API connection failed")
    )

    with patch(
        "app.services.agents.nodes.answer_judge_node.get_ollama_client",
        return_value=mock_client,
    ):
        result = await answer_judge_node(state)

    assert result["answer_judge_attempts"] == 1
    assert result["low_confidence"] is False
    assert result["answer_judge_result"].passed is True
    assert result["answer_judge_feedback"] is None


def test_route_after_answer_judge():
    # Passed
    res1 = AnswerJudgeResult(passed=True, reasoning="OK")
    assert (
        route_after_answer_judge(
            {"answer_judge_result": res1, "answer_judge_attempts": 1}
        )
        == "end"
    )

    # Failed, attempt 1 -> retry
    res2 = AnswerJudgeResult(
        passed=False, reasoning="Missing info", feedback="Add info"
    )
    assert (
        route_after_answer_judge(
            {"answer_judge_result": res2, "answer_judge_attempts": 1}
        )
        == "retry"
    )

    # Failed, attempt 2 -> end
    assert (
        route_after_answer_judge(
            {"answer_judge_result": res2, "answer_judge_attempts": 2}
        )
        == "end"
    )
