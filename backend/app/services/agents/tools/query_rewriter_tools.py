"""
Query Rewriter Tools definitions.
"""

QUERY_REWRITER_TOOLS = [
    {
        "name": "assess_query_quality",
        "description": (
            "Scores the query on clarity, intent completeness, ambiguity, and grammar. "
            "Returns a structured diagnosis including detected issues, confidence score, and suggested rewrite actions. "
            "Returns empty issues and high confidence if the query is already well-formed."
        ),
        "input_schema": {
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
    {
        "name": "rewrite_query",
        "description": (
            "Rewrites the query using a specified strategy based on the diagnosis. "
            "Resolves ambiguity, corrects typos, expands vague intent, and makes implicit constraints explicit. "
            "For follow-up queries, uses conversation history to produce a fully self-contained standalone query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original query to rewrite",
                },
                "strategy": {
                    "type": "string",
                    "description": "One of: disambiguate, correct, expand, both",
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
            "required": ["query", "strategy", "issues", "suggested_actions"],
        },
    },
    {
        "name": "validate_rewrite",
        "description": (
            "Validates that the rewritten query preserves the original intent without adding assumptions the user never stated. "
            "Returns pass/fail verdict with a reason. If failed, the original query should be used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "original_query": {
                    "type": "string",
                    "description": "The original query before rewriting",
                },
                "rewritten_query": {
                    "type": "string",
                    "description": "The rewritten query to validate",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The issues that were being addressed by the rewrite",
                },
            },
            "required": ["original_query", "rewritten_query", "issues"],
        },
    },
]
