from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.services.agents.types import AgentState
from app.services.agents.nodes.orchestrator import orchestrator_node
from app.services.agents.nodes.sql import (
    schema_selection_node,
    sql_generation_node,
)
from app.services.agents.nodes.rag import rag_pipeline_node
from app.services.agents.nodes.fusion import fusion_node


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
        # Safety fallback - should never happen but route to fusion if no agents selected
        next_nodes.append("fusion_node")
    print(f"[Graph] Routing after orchestrator to: {next_nodes}")
    return next_nodes


def route_after_sql_generation(state: AgentState) -> str:
    if state.get("sql_result") and state["sql_result"].success:
        print("[Graph] SQL Agent: success. Routing to fusion.")
        return "sql_done"
    print("[Graph] SQL Agent: failed. Routing to fusion with error.")
    return "sql_failed"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("schema_selection_node", schema_selection_node)
    graph.add_node("sql_generation_node", sql_generation_node)
    graph.add_node("rag_pipeline_node", rag_pipeline_node)
    graph.add_node("fusion_node", fusion_node)

    # Entry point
    graph.add_edge(START, "orchestrator_node")

    # SQL path
    graph.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        {
            "sql_node": "schema_selection_node",
            "rag_pipeline_node": "rag_pipeline_node",
            "fusion_node": "fusion_node",
        },
    )

    graph.add_edge("schema_selection_node", "sql_generation_node")

    graph.add_conditional_edges(
        "sql_generation_node",
        route_after_sql_generation,
        {
            "sql_done": "fusion_node",
            "sql_failed": "fusion_node",
        },
    )

    # RAG path: rag_pipeline_node -> fusion_node
    graph.add_edge("rag_pipeline_node", "fusion_node")

    # Fusion -> END
    graph.add_edge("fusion_node", END)

    return graph


# Compile once at module level - reused across all requests
memory = MemorySaver()
rag_graph = build_graph().compile(checkpointer=memory)
