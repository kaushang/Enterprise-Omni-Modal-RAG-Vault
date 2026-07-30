import inspect
import json

from app.services.agents.ollama_client import OLLAMA_MODEL, get_ollama_client

# ANTHROPIC - restore when switching back to Claude
# from app.services.agents.tools.query_rewriter_tools import (
#     QUERY_REWRITER_TOOL_REGISTRY,
#     QUERY_REWRITER_TOOLS,
# )
from app.services.agents.tools.query_rewriter_tools import (
    QUERY_REWRITER_TOOL_REGISTRY,
    QUERY_REWRITER_TOOLS_OPENAI,
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
            f"{format_tools_for_prompt(QUERY_REWRITER_TOOLS_OPENAI)}\n\n"
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
        # ANTHROPIC - restore when switching back to Claude
        # client = rag_service._get_async_anthropic_client()
        client = get_ollama_client()

        parsed_result = None
        diagnosis_saved = {}

        for turn in range(6):
            # ANTHROPIC - restore when switching back to Claude
            # response = await client.messages.create(
            #     model="claude-haiku-4-5-20251001",
            #     max_tokens=1024,
            #     system=system_prompt,
            #     messages=messages,
            #     tools=QUERY_REWRITER_TOOLS,
            # )
            print("[Query Rewriter] Calling query rewriter agent...")
            response = await client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                tools=QUERY_REWRITER_TOOLS_OPENAI,
                tool_choice="auto",
                max_tokens=1024,
            )

            # ANTHROPIC - restore when switching back to Claude
            # assistant_content = response.content
            # messages.append({"role": "assistant", "content": assistant_content})
            choice = response.choices[0].message
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": choice.tool_calls or [],
                }
            )

            # ANTHROPIC - restore when switching back to Claude
            # text_blocks = [
            #     b.text
            #     for b in assistant_content
            #     if isinstance(getattr(b, "text", None), str) and b.text
            # ]
            text_blocks = [choice.content] if choice.content else []

            # ANTHROPIC - restore when switching back to Claude
            # tool_use_blocks = [
            #     block
            #     for block in assistant_content
            #     if getattr(block, "type", None) == "tool_use"
            # ]
            tool_use_blocks = choice.tool_calls or []

            if not tool_use_blocks:
                full_text = " ".join(text_blocks)
                parsed = parse_json(full_text)
                if parsed and "final_query" in parsed:
                    parsed_result = parsed
                    break
                elif (
                    parsed
                    and "name" in parsed
                    and parsed["name"] in QUERY_REWRITER_TOOL_REGISTRY
                ):
                    # Model returned a tool call as plain text instead of using the function-calling API.
                    # Dispatch it manually without handing back to the LLM for another turn.
                    tool_name = parsed["name"]
                    # Support both "parameters" and "arguments" as the key for tool args
                    tool_args = (
                        parsed.get("parameters") or parsed.get("arguments") or {}
                    )
                    print(
                        f"[Query Rewriter Node] Text-as-tool-call detected: {tool_name}, args: {tool_args}"
                    )
                    tool_fn = QUERY_REWRITER_TOOL_REGISTRY[tool_name]
                    try:
                        res = tool_fn(**tool_args)
                        result = await res if inspect.isawaitable(res) else res
                        if tool_name == "assess_query_quality" and isinstance(
                            result, dict
                        ):
                            diagnosis_saved = result
                            # If assess returned needs_rewrite=True, inject result and let
                            # the loop continue so the agent calls rewrite_query next.
                            if result.get("needs_rewrite"):
                                res_str = json.dumps(result)
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": f"Tool result for {tool_name}: {res_str}",
                                    }
                                )
                                continue
                            # needs_rewrite=False - stop, use original query
                            break
                        elif tool_name == "rewrite_query" and isinstance(result, dict):
                            # Rewrite is done - capture the result directly without
                            # another LLM turn (the LLM was returning None on that turn).
                            rewritten = result.get("rewritten_query", original_query)
                            parsed_result = {
                                "final_query": rewritten,
                                "was_rewritten": rewritten != original_query,
                                "diagnosis": diagnosis_saved,
                            }
                            break
                    except Exception as exc:
                        print(
                            f"[Query Rewriter] Tool execution error ({tool_name}): {exc}"
                        )
                        break
                else:
                    break

            tool_results = []
            for block in tool_use_blocks:
                # ANTHROPIC - restore when switching back to Claude
                # tool_name = block.name
                # tool_args = block.input or {}
                tool_name = block.function.name
                tool_args = json.loads(block.function.arguments or "{}")

                tool_fn = QUERY_REWRITER_TOOL_REGISTRY.get(tool_name)
                if tool_fn:
                    try:
                        print(f"[Query Rewriter] Calling tool: {tool_name}")
                        res = tool_fn(**tool_args)
                        result = await res if inspect.isawaitable(res) else res
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

            # ANTHROPIC - restore when switching back to Claude
            # messages.append({"role": "user", "content": tool_results})
            for tool_result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result["tool_use_id"],
                        "content": tool_result["content"],
                    }
                )

        if parsed_result and "final_query" in parsed_result:
            print(f"[Query Rewriter Node] Parsed result: {parsed_result}")
            res_fq = str(parsed_result["final_query"])
            res_rw = bool(parsed_result.get("was_rewritten", False))
            res_diag = parsed_result.get("diagnosis")
            if not isinstance(res_diag, dict):
                res_diag = diagnosis_saved
            print(f"[Query Rewriter] Final query: {res_fq}, was_rewritten: {res_rw}")
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
