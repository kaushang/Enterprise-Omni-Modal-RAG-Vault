"""
Query Rewriter Tools definitions and implementations.
"""

from app.services.agents.ollama_client import OLLAMA_MODEL, get_ollama_client
from app.services.agents.tools.utils import parse_json

QUERY_REWRITER_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "assess_query_quality",
            "description": (
                "Scores the query on clarity, intent completeness, ambiguity, and grammar. "
                "Returns a structured diagnosis including detected issues, confidence score, and suggested rewrite actions. "
                "Returns empty issues and high confidence if the query is already well-formed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's natural language query",
                    },
                    "conversation_history": {
                        "type": "string",
                        "description": "Recent conversation turns as a JSON string for resolving follow-up queries (optional)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite_query",
            "description": (
                "Rewrites the query using a specified strategy based on the diagnosis. "
                "Resolves ambiguity, corrects typos, expands vague intent, and makes implicit constraints explicit. "
                "For follow-up queries, uses conversation history to produce a fully self-contained standalone query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The original query to rewrite",
                    },
                    "strategies": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "disambiguate",
                                "correct",
                                "expand",
                                "resolve_followup",
                            ],
                        },
                        "description": (
                            "List of rewrite strategies to apply in a single pass. Pick one or more:\n"
                            "- 'disambiguate': resolve vague references, pronouns without referents (them, it, those, same), or queries with multiple possible interpretations\n"
                            "- 'correct': fix typos, spelling errors, and grammatical issues without changing meaning\n"
                            "- 'expand': add missing context, specificity, or implicit constraints the user clearly intended\n"
                            "- 'resolve_followup': use conversation history to rewrite a follow-up query into a fully self-contained standalone question\n"
                        ),
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of detected issues from assess_query_quality",
                    },
                    "suggested_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of suggested actions from assess_query_quality",
                    },
                    "conversation_history": {
                        "type": "string",
                        "description": "Recent conversation turns as a JSON string (optional)",
                    },
                },
                "required": ["query", "strategies", "issues", "suggested_actions"],
            },
        },
    },
]


async def assess_query_quality(query: str = "", conversation_history: str = "") -> dict:
    print(f"[Query Rewriter - assess_query_quality] Assessing query: {query}")
    try:
        # ANTHROPIC - restore when switching back to Claude
        # client = rag_service._get_async_anthropic_client()
        # response = await client.messages.create(
        #     model="claude-haiku-4-5-20251001",
        #     max_tokens=512,
        #     system=system_prompt,
        #     messages=[{"role": "user", "content": user_prompt}],
        # )
        # raw_text = response.content[0].text if response.content else ""

        client = get_ollama_client()
        system_prompt = (
            "You are a query quality assessor. Return ONLY a JSON object with these keys:\n"
            'issues: list from ["vague_entity","ambiguous_intent","typo","follow_up_without_context"]\n'
            "confidence: float 0.0-1.0 (1.0=perfectly clear)\n"
            'suggested_actions: list from ["correct_spelling","expand","disambiguate","resolve_followup","specify_entity"]\n'
            "needs_rewrite: true if confidence<0.7 or issues non-empty\n\n"
            "follow_up_without_context rules:\n"
            "- Flag if query has pronouns/references with no in-query referent (them,they,it,those,same,that,these)\n"
            "- Flag if query starts with: and,but,also,what about,same for\n"
            "- Never flag a complete query with explicit subject and goal\n"
            "- Follow-up: 'which of them performed best?', 'what about last month?'\n"
            "- Not follow-up: 'Give a summary of orders per product category'\n"
            "No markdown. No text outside JSON."
        )

        user_prompt = f"Query: {query}\n"
        if conversation_history:
            user_prompt += f"Conversation History:\n{conversation_history}\n"
        user_prompt += "Assess whether this query needs rewriting. Only flag real problems, not hypothetical ones."

        response = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
        )

        raw_text = response.choices[0].message.content or ""

        parsed = parse_json(raw_text)
        print(f"[Query Rewriter - assess_query_quality] Result: {parsed}")
        if parsed and isinstance(parsed, dict):
            issues = list(parsed.get("issues", []))
            confidence = float(parsed.get("confidence", 1.0))
            suggested_actions = list(parsed.get("suggested_actions", []))
            # Enforce the rule in Python - do not trust the model's needs_rewrite value.
            # Raised threshold from 0.85 to 0.7 to avoid rewriting clear queries the model
            # is merely uncertain about. Only rewrite on real detected issues (confidence < 0.7)
            # or explicit structural problems like typos and unresolved follow-ups.
            needs_rewrite = bool(issues) and confidence < 0.7
            print(
                f"[Query Rewriter - assess_query_quality] Diagnosis: needs_rewrite={needs_rewrite}, confidence={confidence}, issues={issues}, suggested actions={suggested_actions}"
            )
            if not needs_rewrite:
                print(
                    "[Query Rewriter - assess_query_quality] Query unchanged - already well-formed"
                )
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
    query: str = "",
    strategies: list[str] | None = None,
    issues: list | None = None,
    suggested_actions: list | None = None,
    conversation_history: str = "",
) -> dict:
    print(f"[Query Rewriter - rewrite_query] Rewriting with strategy: {strategies}")
    strategies = strategies or ["expand"]
    issues = issues or []
    suggested_actions = suggested_actions or []
    try:
        # ANTHROPIC - restore when switching back to Claude
        # client = rag_service._get_async_anthropic_client()
        # response = await client.messages.create(
        #     model="claude-haiku-4-5-20251001",
        #     max_tokens=512,
        #     system=system_prompt,
        #     messages=[{"role": "user", "content": user_prompt}],
        # )
        # rewritten = response.content[0].text.strip() if response.content else query

        client = get_ollama_client()
        system_prompt = (
            "You are a query rewriter. Your job is to improve the given query based on the detected issues and strategy. Rules:\n\n"
            "Never add assumptions the user did not state\n"
            "Never change the core intent of the query, if the user asks more than 1 query, all of them should be addressed\n"
            "For follow-up queries, use conversation history to make the query fully self-contained\n"
            "For typo correction, fix spelling and grammar only, do not change meaning\n"
            "For disambiguation, make implicit constraints explicit based only on what can be reasonably inferred\n"
            "Return ONLY the rewritten query as plain text - no explanation, no preamble, no quotes"
        )

        user_prompt = (
            f"Original Query: {query}\n"
            f"Strategies: {strategies}\n"
            f"Issues: {issues}\n"
            f"Suggested Actions: {suggested_actions}\n"
        )
        if conversation_history:
            user_prompt += f"Conversation History:\n{conversation_history}\n"
        user_prompt += "Rewrite the query following the rules."

        response = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
        )

        rewritten = (
            response.choices[0].message.content.strip()
            if response.choices[0].message and response.choices[0].message.content
            else query
        )
        if (rewritten.startswith('"') and rewritten.endswith('"')) or (
            rewritten.startswith("'") and rewritten.endswith("'")
        ):
            rewritten = rewritten[1:-1].strip()
        rewritten = rewritten if rewritten else query
        print("[Query Rewriter - rewrite_query] Rewritten Query: ", rewritten)
        return {"rewritten_query": rewritten}
    except Exception as e:
        print(f"[Query Rewriter] Exception in rewrite_query: {e}")
        return {"rewritten_query": query}


QUERY_REWRITER_TOOL_REGISTRY: dict[str, callable] = {
    "assess_query_quality": assess_query_quality,
    "rewrite_query": rewrite_query,
}
