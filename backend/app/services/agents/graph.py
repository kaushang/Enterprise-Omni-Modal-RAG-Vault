from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.services.agents.nodes.answer_generation_node import answer_generation_node
from app.services.agents.nodes.answer_judge_node import answer_judge_node
from app.services.agents.nodes.orchestrator import orchestrator_node
from app.services.agents.nodes.query_rewriter import query_rewriter_node
from app.services.agents.nodes.rag import rag_pipeline_node
from app.services.agents.nodes.sql import (
    schema_selection_node,
    sql_generation_node,
    sql_judge_node,
)
from app.services.agents.types import AgentState


def route_after_orchestrator(state: AgentState) -> list[str]:
    """
    After orchestrator decides the plan, route to the correct agent nodes.
    Returns a list of node names to invoke next (parallel execution if multiple).
    """
    next_nodes = []
    if state["invoke_sql"]:
        next_nodes.append("sql_node")
    if state["invoke_rag"]:
        next_nodes.append("rag_pipeline_node")
    if not next_nodes:
        # Safety fallback - should never happen but route to answer_generation_node if no agents selected
        next_nodes.append("answer_generation_node")
    print(f"[Graph] Routing after orchestrator to: {next_nodes}")
    return next_nodes


def route_after_sql_judge(state: AgentState) -> str:
    if state.get("sql_generation_error"):
        print(
            "[Graph] SQL generation failed. Skipping judge, routing to answer_generation_node."
        )
        return "sql_failed"

    judge_result = state.get("judge_result")
    passed = False
    critical_hints = []
    if judge_result:
        passed = (
            getattr(judge_result, "passed", False)
            if hasattr(judge_result, "passed")
            else (
                judge_result.get("passed", False)
                if isinstance(judge_result, dict)
                else False
            )
        )
        critical_hints = (
            getattr(judge_result, "critical_optimization_hints", [])
            if hasattr(judge_result, "critical_optimization_hints")
            else (
                judge_result.get("critical_optimization_hints", [])
                if isinstance(judge_result, dict)
                else []
            )
        )

    if passed and not critical_hints:
        print(
            "[Graph] SQL Judge approved query with no critical optimization issues. Routing to answer_generation_node."
        )
        return "sql_approved"

    retry_count = state.get("sql_judge_retry_count", 0)
    if retry_count > 1:
        print(
            f"[Graph] SQL Judge requested retry but max retries (1) reached (retry_count={retry_count}). "
            "Routing to answer_generation_node with best available SQL."
        )
        return "sql_approved"

    print(
        f"[Graph] SQL Judge requested retry (passed={passed}, critical_hints={critical_hints}, retry_count={retry_count}). "
        "Routing to sql_generation_node for retry."
    )
    return "sql_retry"


def _sql_returned_empty(state: AgentState) -> bool:
    """Returns True if SQL was the only source, executed successfully, and returned no rows."""
    sql_result = state.get("sql_result")
    rag_result = state.get("rag_result")
    sql_ok = bool(sql_result and sql_result.success)
    sql_empty = sql_ok and not sql_result.query_results
    rag_ok = bool(rag_result and rag_result.success)
    return sql_empty and not rag_ok


def route_after_answer_generation(state: AgentState) -> str:
    """
    Decide whether to run the answer judge or go straight to END.
    Skip the judge when the answer was short-circuited due to empty SQL results -
    the answer is deterministic in that case and there is nothing to judge.
    """
    if _sql_returned_empty(state):
        print("[Graph] SQL returned 0 rows - skipping answer judge, routing to END.")
        return "skip_judge"
    return "judge"


def route_after_answer_judge(state: AgentState) -> str:
    judge_res = state.get("answer_judge_result")
    attempts = state.get("answer_judge_attempts", 0)

    if not judge_res:
        return "end"

    passed = (
        getattr(judge_res, "passed", True)
        if hasattr(judge_res, "passed")
        else (judge_res.get("passed", True) if isinstance(judge_res, dict) else True)
    )

    if passed:
        return "end"

    if attempts < 2:
        return "retry"

    return "end"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("query_rewriter_node", query_rewriter_node)
    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("schema_selection_node", schema_selection_node)
    graph.add_node("sql_generation_node", sql_generation_node)
    graph.add_node("sql_judge_node", sql_judge_node)
    graph.add_node("rag_pipeline_node", rag_pipeline_node)
    graph.add_node("answer_generation_node", answer_generation_node)
    graph.add_node("answer_judge_node", answer_judge_node)

    # Entry point
    graph.add_edge(START, "query_rewriter_node")
    graph.add_edge("query_rewriter_node", "orchestrator_node")

    # SQL path
    graph.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        {
            "sql_node": "schema_selection_node",
            "rag_pipeline_node": "rag_pipeline_node",
            "answer_generation_node": "answer_generation_node",
        },
    )

    graph.add_edge("schema_selection_node", "sql_generation_node")
    graph.add_edge("sql_generation_node", "sql_judge_node")

    graph.add_conditional_edges(
        "sql_judge_node",
        route_after_sql_judge,
        {
            "sql_approved": "answer_generation_node",
            "sql_failed": "answer_generation_node",
            "sql_retry": "sql_generation_node",
        },
    )

    # RAG path
    graph.add_edge("rag_pipeline_node", "answer_generation_node")

    # Answer Generation -> conditionally route to judge or END
    graph.add_conditional_edges(
        "answer_generation_node",
        route_after_answer_generation,
        {
            "judge": "answer_judge_node",
            "skip_judge": END,
        },
    )

    # Route after answer judge: END or retry answer_generation_node
    graph.add_conditional_edges(
        "answer_judge_node",
        route_after_answer_judge,
        {
            "end": END,
            "retry": "answer_generation_node",
        },
    )

    return graph


# Compile once at module level - reused across all requests
memory = MemorySaver()
rag_graph = build_graph().compile(checkpointer=memory)
