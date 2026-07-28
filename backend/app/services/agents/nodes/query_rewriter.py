import json  # noqa: I001

import app.services.rag_service as rag_service  
from app.services.agents.tools.query_rewriter_tools import QUERY_REWRITER_TOOLS  
from app.services.agents.tools.utils import format_tools_for_prompt, parse_json
from app.services.agents.types import AgentState


async def assess_query_quality(query: str, conversation_history: str = "") -> dict:
    try:
        client = rag_service._get_async_anthropic_client()
        # In assess_query_quality system prompt - replace the existing one
        system_prompt = (
            "You are a query quality assessor. Analyze the given query and return ONLY a JSON object with these exact keys:\n\n"
            'issues: list of strings from: ["missing_timeframe", "vague_entity", "ambiguous_intent", "typo", "missing_scope", "follow_up_without_context", "too_broad", "missing_metric"]\n'
            "confidence: float 0.0-1.0 (1.0 = perfectly clear and complete)\n"
            'suggested_actions: list of strings from: ["correct_spelling", "expand", "disambiguate", "resolve_followup", "add_timeframe", "specify_entity", "specify_metric"]\n'
            "needs_rewrite: boolean - true only if confidence < 0.85 or issues is non-empty\n\n"
            "Critical rules for follow_up_without_context:\n"
            "- Only flag this if the query contains pronouns without referents (it, they, that, same, those), "
            "or words like 'also', 'and what about', 'same for', or is clearly missing a subject.\n"
            "- A complete imperative query with a clear subject and goal is NEVER a follow-up.\n"
            "- Example of follow-up: 'what about last month?' or 'show me the same for north region'\n"
            "- Example of NOT a follow-up: 'Give a summary of orders per product category'\n\n"
            "No markdown, no explanation outside the JSON."
        )

        user_prompt = f"Query: {query}\n"
        if conversation_history:
            user_prompt += f"Conversation History:\n{conversation_history}\n"
        user_prompt += "Assess whether this is a standalone clear query or a follow-up that needs context resolution."

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text if response.content else ""
        parsed = parse_json(raw_text)
        if parsed and isinstance(parsed, dict):
            issues = list(parsed.get("issues", []))
            confidence = float(parsed.get("confidence", 1.0))
            suggested_actions = list(parsed.get("suggested_actions", []))
            needs_rewrite = bool(parsed.get("needs_rewrite", False))
            return {
                "issues": issues,
                "confidence": confidence,
                "suggested_actions": suggested_actions,
                "needs_rewrite": needs_rewrite,
            }
    except Exception as e:
        print(f"[Query Rewriter] Exception in assess_query_quality: {e}")

    return {
        "issues": [],
        "confidence": 1.0,
        "suggested_actions": [],
        "needs_rewrite": False,
    }


async def rewrite_query(
    query: str,
    strategy: str,
    issues: list,
    suggested_actions: list,
    conversation_history: str = "",
) -> str:
    try:
        client = rag_service._get_async_anthropic_client()
        system_prompt = (
            "You are a query rewriter. Your job is to improve the given query based on the detected issues and strategy. Rules:\n\n"
            "Never add assumptions the user did not state\n"
            "Never change the core intent of the query\n"
            "For follow-up queries, use conversation history to make the query fully self-contained\n"
            "For typo correction, fix spelling and grammar only, do not change meaning\n"
            "For disambiguation, make implicit constraints explicit based only on what can be reasonably inferred\n"
            "Return ONLY the rewritten query as plain text - no explanation, no preamble, no quotes"
        )

        user_prompt = (
            f"Original Query: {query}\n"
            f"Strategy: {strategy}\n"
            f"Issues: {issues}\n"
            f"Suggested Actions: {suggested_actions}\n"
        )
        if conversation_history:
            user_prompt += f"Conversation History:\n{conversation_history}\n"
        user_prompt += "Rewrite the query following the rules."

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        rewritten = response.content[0].text.strip() if response.content else query
        if (rewritten.startswith('"') and rewritten.endswith('"')) or (
            rewritten.startswith("'") and rewritten.endswith("'")
        ):
            rewritten = rewritten[1:-1].strip()
        return rewritten if rewritten else query
    except Exception as e:
        print(f"[Query Rewriter] Exception in rewrite_query: {e}")
        return query


async def validate_rewrite(
    original_query: str, rewritten_query: str, issues: list
) -> tuple[bool, str]:
    try:
        client = rag_service._get_async_anthropic_client()
        system_prompt = (
            "You are a rewrite validator. Check whether the rewritten query preserves the original intent without adding unstated assumptions. Return ONLY a JSON object with:\n\n"
            "valid: boolean\n"
            "reason: string explaining why it passed or failed"
        )

        user_prompt = (
            f"Original Query: {original_query}\n"
            f"Rewritten Query: {rewritten_query}\n"
            f"Issues Addressed: {issues}\n"
            "Validate whether the rewritten query preserves original intent without adding unstated assumptions."
        )

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text if response.content else ""
        parsed = parse_json(raw_text)
        if parsed and isinstance(parsed, dict):
            valid = bool(parsed.get("valid", True))
            reason = str(parsed.get("reason", "Validation completed"))
            return valid, reason
    except Exception as e:
        print(f"[Query Rewriter] Exception in validate_rewrite: {e}")

    return True, "Validation skipped due to error"


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
            "- Always call validate_rewrite after rewriting. If validation fails, use the original query.\n"
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
        final_query = original_query
        was_rewritten = False

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

                if tool_name == "assess_query_quality":
                    q = tool_args.get("query") or query
                    h = tool_args.get("conversation_history") or history_str
                    print(f"[Query Rewriter] Assessing query: {q}")
                    diag = await assess_query_quality(q, h)
                    diagnosis_saved = diag
                    needs_rw = diag.get("needs_rewrite", False)
                    conf = diag.get("confidence", 1.0)
                    iss = diag.get("issues", [])
                    print(
                        f"[Query Rewriter] Diagnosis: needs_rewrite={needs_rw}, confidence={conf}, issues={iss}"
                    )
                    if not needs_rw:
                        print("[Query Rewriter] Query unchanged - already well-formed")
                    res_str = json.dumps(diag)

                elif tool_name == "rewrite_query":
                    q = tool_args.get("query") or query
                    strat = tool_args.get("strategy") or "expand"
                    iss = tool_args.get("issues") or []
                    act = tool_args.get("suggested_actions") or []
                    h = tool_args.get("conversation_history") or history_str
                    print(f"[Query Rewriter] Rewriting with strategy: {strat}")
                    rewritten = await rewrite_query(q, strat, iss, act, h)
                    res_str = json.dumps({"rewritten_query": rewritten})

                elif tool_name == "validate_rewrite":
                    orig = tool_args.get("original_query") or original_query
                    rewr = tool_args.get("rewritten_query") or ""
                    iss = tool_args.get("issues") or []
                    valid, reason = await validate_rewrite(orig, rewr, iss)
                    fq = rewr if valid else orig
                    if valid:
                        was_rewritten = fq != orig
                        final_query = fq
                    # In validate_rewrite dispatch - add reason to log
                    print(
                        f"[Query Rewriter] Rewrite validated: {valid}. Reason: {reason}. Final query: {fq}"
                    )
                    res_str = json.dumps(
                        {"valid": valid, "reason": reason, "final_query": fq}
                    )

                else:
                    res_str = json.dumps({"error": f"Unknown tool {tool_name}"})

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
            "query": final_query,
            "original_query": original_query,
            "query_was_rewritten": was_rewritten,
            "rewrite_diagnosis": diagnosis_saved,
        }

    except Exception as e:
        print(f"[Query Rewriter] Error in query_rewriter_node: {e}")
        return {"query": original_query}
