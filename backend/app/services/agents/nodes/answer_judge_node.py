import logging

# from app.services.agents.ollama_client import OLLAMA_MODEL, get_ollama_client
from app.services.agents.tools.utils import parse_json
from app.services.agents.types import AgentState, AnswerJudgeResult
import app.services.rag_service as rag_service

logger = logging.getLogger(__name__)


async def answer_judge_node(state: AgentState) -> dict:
    attempts = state.get("answer_judge_attempts", 0) + 1
    query = state.get("original_query") or state.get("query", "")
    final_answer = state.get("final_answer", "")

    print(f"[Answer Judge] Attempt {attempts} | Query: {query[:120]}")
    print(
        f"[Answer Judge] Final answer length: {len(final_answer)} chars | Preview: {final_answer[:200]}"
    )

    if not final_answer:
        print("[Answer Judge] No final answer found - returning failed result")
        judge_res = AnswerJudgeResult(
            passed=False,
            reasoning="No final answer was synthesized by the answer generation node.",
            feedback="The generated answer is empty.",
        )
        return {
            "answer_judge_result": judge_res,
            "answer_judge_feedback": judge_res.feedback,
            "answer_judge_attempts": attempts,
            "low_confidence": False,
        }

    context_parts = []
    if state.get("rag_result"):
        context_parts.append(f"Document context:\n{state['rag_result']}")
    if state.get("sql_result"):
        context_parts.append(f"SQL result:\n{state['sql_result']}")
    context_str = (
        "\n\n".join(context_parts)
        if context_parts
        else "No retrieved context available."
    )

    print(
        f"[Answer Judge] Context sources - rag_result: {bool(state.get('rag_result'))}, sql_result: {bool(state.get('sql_result'))}"
    )

    system_prompt = (
        "You are an Answer Judge Agent in a RAG pipeline.\n"
        "You evaluate whether a synthesized answer properly addresses the user's query and is grounded in the retrieved context.\n\n"
        "Evaluate the following three things:\n"
        "1. Grounding - identify any claims in the answer that are NOT supported by the retrieved context. List each unsupported claim.\n"
        "2. Completeness - identify any parts of the user query that the answer silently skipped or did not address. List each missed intent.\n"
        "3. Overall pass/fail - passed is true ONLY if grounding_issues is empty AND missing_intents is empty.\n\n"
        "Return ONLY a valid JSON object with no markdown, no commentary:\n"
        "{\n"
        '    "passed": true,\n'
        '    "reasoning": "...",\n'
        '    "grounding_issues": [],\n'
        '    "missing_intents": [],\n'
        '    "feedback": null\n'
        "}\n\n"
        "feedback must be a detailed actionable string when passed is false, telling the synthesis node exactly what to fix. feedback is null when passed is true."
    )

    user_prompt = (
        f"User Query:\n{query}\n\n"
        f"Retrieved Context:\n{context_str}\n\n"
        f"Synthesized Answer:\n{final_answer}\n\n"
        "Evaluate the answer and return JSON."
    )

    try:
        # ANTHROPIC - restore when switching back to Claude
        client = rag_service._get_async_anthropic_client()
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                raw_text += block.text

        # client = get_ollama_client()
        # response = await client.chat.completions.create(
        #     model=OLLAMA_MODEL,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     max_tokens=1024,
        # )
        # raw_text = response.choices[0].message.content or ""

        print(f"[Answer Judge] Raw LLM response: {raw_text}")

        parsed_data = parse_json(raw_text)

        print(f"[Answer Judge] Parsed result: {parsed_data}")

        if not parsed_data or not isinstance(parsed_data, dict):
            raise ValueError("Invalid or non-dict JSON response from Answer Judge LLM")

        passed = bool(parsed_data.get("passed", False))
        reasoning = str(parsed_data.get("reasoning", ""))
        grounding_issues = list(parsed_data.get("grounding_issues", []))
        missing_intents = list(parsed_data.get("missing_intents", []))
        feedback = parsed_data.get("feedback")
        if feedback is not None:
            feedback = str(feedback)

        print(
            f"[Answer Judge] passed={passed} | grounding_issues={grounding_issues} | missing_intents={missing_intents}"
        )
        print(f"[Answer Judge] Feedback: {feedback}")

        # After parsing, if attempts >= 2 and passed is False:
        if attempts >= 2 and not passed:
            logger.warning(
                "[Answer Judge Node] Max attempts reached. Passing answer with low_confidence=True."
            )
            print(
                f"[Answer Judge] Max attempts ({attempts}) reached with passed=False - forcing low_confidence pass"
            )
            judge_res = AnswerJudgeResult(
                passed=True,
                reasoning=reasoning,
                grounding_issues=grounding_issues,
                missing_intents=missing_intents,
                feedback=feedback,
                low_confidence=True,
            )
            return {
                "answer_judge_result": judge_res,
                "answer_judge_feedback": None,
                "answer_judge_attempts": attempts,
                "low_confidence": True,
            }

        print(
            f"[Answer Judge] Returning result - passed={passed}, low_confidence=False"
        )
        judge_res = AnswerJudgeResult(
            passed=passed,
            reasoning=reasoning,
            grounding_issues=grounding_issues,
            missing_intents=missing_intents,
            feedback=feedback,
            low_confidence=False,
        )
        return {
            "answer_judge_result": judge_res,
            "answer_judge_feedback": feedback if not passed else None,
            "answer_judge_attempts": attempts,
            "low_confidence": False,
        }

    except Exception as e:
        logger.error(f"[Answer Judge Node] Soft fallback triggered due to error: {e}")
        print(f"[Answer Judge] Exception caught - soft fallback triggered: {e}")
        judge_res = AnswerJudgeResult(
            passed=True,
            reasoning=f"Soft fallback due to error: {e}",
            grounding_issues=[],
            missing_intents=[],
            feedback=None,
            low_confidence=False,
        )
        return {
            "answer_judge_result": judge_res,
            "answer_judge_feedback": None,
            "answer_judge_attempts": attempts,
            "low_confidence": False,
        }
