import json

import app.services.rag_service as rag_service
from app.services.agents.tools.query_rewriter_tools import (
    QUERY_REWRITER_TOOL_REGISTRY,
    QUERY_REWRITER_TOOLS,
)
from app.services.agents.tools.utils import format_tools_for_prompt, parse_json
from app.services.agents.types import AgentState


async def query_rewriter_node(state: AgentState) -> dict:
    try:
        query = state["query"]
        turns = state.get("db_session_turns") or []
        history_str = json.dumps(turns, default=str)
        original_query = query

        system_prompt = (
            "You are a Query Rewriter Agent operating in a ReAct loop. Your goal is to assess whether the user's query needs improvement and rewrite it if needed.\n\n"
            f"{format_tools_for_prompt(QUERY_REWRITER_TOOLS)}\n\n"
            "Rules:\n"
            "- Always call assess_query_quality first.\n"
            "- If needs_rewrite is false, stop immediately - do not call any other tools.\n"
            "- If needs_rewrite is true, call rewrite_query with the appropriate strategy derived from suggested_actions.\n"
            "- Never rewrite a query that is already clear and complete.\n"
            "- End your turn by returning ONLY a JSON object:\n"
            "{\n"
            '    "final_query": "the rewritten or original query",\n'
            '    "was_rewritten": true or false,\n'
            '    "diagnosis": {...diagnosis from assess_query_quality...}\n'
            "}"
        )

        user_prompt = (
            f"Original Query: {original_query}\nConversation History: {history_str}"
        )
        messages = [{"role": "user", "content": user_prompt}]
        client = rag_service._get_async_anthropic_client()

        parsed_result = None
        diagnosis_saved = {}

        for turn in range(6):
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=QUERY_REWRITER_TOOLS,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            text_blocks = [
                b.text
                for b in assistant_content
                if isinstance(getattr(b, "text", None), str) and b.text
            ]

            tool_use_blocks = [
                block
                for block in assistant_content
                if getattr(block, "type", None) == "tool_use"
            ]

            if not tool_use_blocks:
                full_text = " ".join(text_blocks)
                parsed = parse_json(full_text)
                if parsed and "final_query" in parsed:
                    parsed_result = parsed
                    break
                else:
                    break

            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input or {}

                tool_fn = QUERY_REWRITER_TOOL_REGISTRY.get(tool_name)
                if tool_fn:
                    try:
                        result = await tool_fn(**tool_args)
                        if tool_name == "assess_query_quality" and isinstance(
                            result, dict
                        ):
                            diagnosis_saved = result
                        res_str = json.dumps(
                            result if isinstance(result, dict) else {"result": result}
                        )
                    except Exception as exc:
                        print(
                            f"[Query Rewriter] Tool execution error ({tool_name}): {exc}"
                        )
                        res_str = json.dumps({"error": str(exc)})
                else:
                    res_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": res_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        if parsed_result and "final_query" in parsed_result:
            res_fq = str(parsed_result["final_query"])
            res_rw = bool(parsed_result.get("was_rewritten", False))
            res_diag = parsed_result.get("diagnosis")
            if not isinstance(res_diag, dict):
                res_diag = diagnosis_saved
            return {
                "query": res_fq,
                "original_query": original_query,
                "query_was_rewritten": res_rw,
                "rewrite_diagnosis": res_diag,
            }

        return {
            "query": original_query,
            "original_query": original_query,
            "query_was_rewritten": False,
            "rewrite_diagnosis": diagnosis_saved,
        }

    except Exception as e:
        print(f"[Query Rewriter] Error in query_rewriter_node: {e}")
        return {"query": original_query}
