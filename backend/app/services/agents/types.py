from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Literal, Optional, TypedDict


class AgentState(TypedDict):
    # --- Input fields (set once at graph entry, never modified) ---
    query: str
    user_id: str
    tenant_id: str
    database_id: Optional[str]
    document_id: Optional[str]
    conversation_history: str
    model_id: Optional[str]
    session_id: Optional[str]
    command_instruction: Optional[str]
    compare_document_ids: Optional[list[str]]
    is_compare_mode: bool
    is_summarize_mode: bool

    # --- Orchestrator output ---
    invoke_sql: bool
    invoke_rag: bool
    mode: Literal["db_only", "doc_only", "cross_source"]
    orchestrator_reasoning: str

    # --- Progress messages streamed to user ---
    # Uses add_messages reducer so each node appends rather than overwrites
    progress_tokens: Annotated[list[str], operator.add]

    # --- Agent results ---
    sql_result: Optional[SQLAgentResult]
    rag_result: Optional[RAGAgentResult]

    # --- Context gathering (set once, never recomputed) ---
    db_user_id: Optional[str]
    db_connection_id: Optional[str]
    db_authorized_schema: Optional[dict]
    db_session_turns: Optional[list]
    db_authorized_cols_by_table: Optional[dict]
    db_all_physical_cols_by_table: Optional[dict]
    db_valid_tables: Optional[set]
    db_connection_engine: Optional[str]
    db_connection_name: Optional[str]
    db_is_admin: bool
    context_error: Optional[str]

    # --- Query Rewriter output ---
    original_query: str
    query_was_rewritten: bool
    rewrite_diagnosis: dict

    # --- Query understanding ---
    query_plan: Optional[dict]

    # --- Schema selection ---
    db_filtered_schema: Optional[dict]

    # --- SQL generation & Judge ---
    generated_sql: Optional[str]
    previous_sql: Optional[str]
    sql_generation_attempts: int
    sql_generation_error: Optional[str]
    judge_result: Optional[JudgeResult]
    sql_judge_approved: bool
    sql_judge_retry_count: int

    # --- Shared RAG Pipeline State ---
    collection_name: Optional[str]
    doc_id_to_filename: Optional[dict[str, str]]
    search_role_ids: Optional[list[str]]
    tenant_id: Optional[str]
    authorized_doc_ids: Optional[list[str]]

    # --- Answer Generation output ---
    final_answer: str
    citations: list[dict]
    follow_up_questions: list[str]
    chart_spec: Optional[dict]
    query_results: Optional[list[dict]]
    model_string: Optional[str]
    resolved_model: Optional[str]
    resolved_model_id: Optional[str]
    was_fallback: bool
    fallback_model_name: Optional[str]
    execution_time_ms: int

    # --- Answer Judge output ---
    answer_judge_result: Optional[AnswerJudgeResult]
    answer_judge_feedback: Optional[str]
    answer_judge_attempts: int
    low_confidence: bool


# --- Agent result types ---


@dataclass
class SQLAgentResult:
    success: bool
    sql_query: Optional[str] = None
    query_results: Optional[list[dict]] = None
    formatted_results: Optional[str] = None
    connection_name: Optional[str] = None
    connection_id: Optional[uuid.UUID] = None
    execution_time_ms: int = 0
    confidence: float = 0.0  # 0.0 to 1.0
    reasoning: str = ""  # agent's own explanation of its output quality
    attempts: int = 1  # how many ReAct iterations were needed
    error: Optional[str] = None


@dataclass
class JudgeResult:
    passed: bool
    semantic_score: float
    failed_filters: list[str]
    critical_optimization_hints: list[str]
    optimization_hints: list[str]
    retry_feedback: str


@dataclass
class RAGAgentResult:
    success: bool
    qdrant_results: list[dict] = field(default_factory=list)
    excel_results: list[dict] = field(default_factory=list)
    context_block: str = "No relevant context found."
    doc_id_to_filename: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    attempts: int = 1
    reformulated_query: Optional[str] = (
        None  # if agent reformulated the query during retry
    )


@dataclass
class AnswerJudgeResult:
    passed: bool
    reasoning: str
    grounding_issues: list[str] = field(default_factory=list)
    missing_intents: list[str] = field(default_factory=list)
    feedback: Optional[str] = None
    low_confidence: bool = False


if __name__ == "__main__":
    state = AgentState(
        query="test",
        user_id="test",
        tenant_id="test",
        database_id=None,
        document_id=None,
        conversation_history="",
        model_id=None,
        session_id=None,
        command_instruction=None,
        compare_document_ids=None,
        is_compare_mode=False,
        is_summarize_mode=False,
        invoke_sql=False,
        invoke_rag=False,
        mode="doc_only",
        orchestrator_reasoning="",
        progress_tokens=[],
        sql_result=None,
        rag_result=None,
        db_user_id=None,
        db_connection_id=None,
        db_authorized_schema=None,
        db_session_turns=None,
        db_authorized_cols_by_table=None,
        db_all_physical_cols_by_table=None,
        db_valid_tables=None,
        db_connection_engine=None,
        db_connection_name=None,
        db_is_admin=False,
        context_error=None,
        original_query="test",
        query_was_rewritten=False,
        rewrite_diagnosis={},
        query_plan=None,
        db_filtered_schema=None,
        generated_sql=None,
        previous_sql=None,
        sql_generation_attempts=0,
        sql_generation_error=None,
        judge_result=None,
        sql_judge_approved=False,
        sql_judge_retry_count=0,
        collection_name=None,
        doc_id_to_filename=None,
        search_role_ids=None,
        authorized_doc_ids=None,
        final_answer="",
        citations=[],
        follow_up_questions=[],
        chart_spec=None,
        query_results=None,
        model_string=None,
        resolved_model=None,
        resolved_model_id=None,
        was_fallback=False,
        fallback_model_name=None,
        execution_time_ms=0,
        answer_judge_result=None,
        answer_judge_feedback=None,
        answer_judge_attempts=0,
        low_confidence=False,
    )
    print("AgentState OK")
    print("LangGraph import OK")
