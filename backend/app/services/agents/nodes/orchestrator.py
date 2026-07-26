# Orchestrator node - routing logic

from app.services.agents.types import AgentState


async def orchestrator_node(state: AgentState) -> dict:
    db_id = state.get("database_id")
    doc_id = state.get("document_id")

    print(f"[Orchestrator] Query: {state['query'][:100]}")
    print(f"[Orchestrator] database_id={db_id}, document_id={doc_id}")

    if db_id and doc_id:
        plan = {
            "invoke_sql": True,
            "invoke_rag": True,
            "mode": "cross_source",
            "reasoning": "Both database and document sources are provided. Routing to cross-source query pipeline.",
        }
    elif db_id:
        plan = {
            "invoke_sql": True,
            "invoke_rag": False,
            "mode": "db_only",
            "reasoning": "Database source is provided, but document source is not. Routing to SQL database pipeline.",
        }
    else:
        plan = {
            "invoke_sql": False,
            "invoke_rag": True,
            "mode": "doc_only",
            "reasoning": "No database source provided. Defaulting to RAG semantic search pipeline.",
        }

    print(
        f"[Orchestrator] Plan: mode={plan['mode']}, invoke_sql={plan['invoke_sql']}, invoke_rag={plan['invoke_rag']}"
    )
    print(f"[Orchestrator] Reasoning: {plan['reasoning']}")

    return {
        "invoke_sql": plan["invoke_sql"],
        "invoke_rag": plan["invoke_rag"],
        "mode": plan["mode"],
        "orchestrator_reasoning": plan["reasoning"],
        "sql_generation_attempts": 0,
        "progress_tokens": ["*Analyzing query and planning agent workflow...*\n\n"],
    }
