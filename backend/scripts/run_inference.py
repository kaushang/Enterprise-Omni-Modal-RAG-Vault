"""
Step 0 Findings:
----------------
1. LangGraph State Schema & Final Answer Key:
   - State Schema: `AgentState` (TypedDict defined in `app/services/agents/types.py`).
   - Final Answer Key: `"final_answer"` (str) in `AgentState`. Populated by `answer_generation_node`.

2. Retrieved Context Chunks Key:
   - State Key: `"rag_result"` (`RAGAgentResult` dataclass in `AgentState`) containing:
     - `qdrant_results`: list of hit dicts where payload holds `"chunk_text"`.
     - `excel_results`: list of dicts where item holds `"result"`.
   - Also mirrored in `"citations"` (list of dicts with `"chunk_text"`) populated by `answer_generation_node`.

3. Where rag_graph is Defined and Imported:
   - Defined in: `backend/app/services/agents/graph.py` (compiled via `rag_graph = build_graph().compile(checkpointer=memory)`).
   - Imported via: `from app.services.agents.graph import rag_graph`.

4. Construction of initial_state in FastAPI Route:
   - In `app/services/rag_service.py` (`run_rag_pipeline`) and `app/api/v1/chat.py`:
     `initial_state` is an `AgentState` TypedDict initialized with:
     - `query`: user question string
     - `user_id`: string UUID of the user
     - `tenant_id`: string UUID of the tenant
     - `database_id`: None (or string UUID if attached)
     - `document_id`: None (or string UUID if attached)
     - `conversation_history`: "" (or formatted history string)
     - `model_id`: None (or model UUID/auto)
     - `session_id`: string UUID for the session
     - `command_instruction`: None
     - `compare_document_ids`: None
     - `is_compare_mode`: False
     - `is_summarize_mode`: False
     - Node flags and outputs initialized to defaults: `invoke_sql=False`, `invoke_rag=False`,
       `mode="doc_only"`, `progress_tokens=[]`, `sql_result=None`, `rag_result=None`,
       `final_answer=""`, `citations=[]`, `follow_up_questions=[]`, etc.

5. Construction of config for rag_graph.astream():
   - In `app/services/rag_service.py`:
     `config = {"configurable": {"thread_id": str(session_id)}}`
     where `thread_id` is a unique session/execution UUID string passed to MemorySaver.
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv

# Ensure the backend directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env
load_dotenv(find_dotenv())

from app.db.session import SessionLocal
from app.models.ragas import (
    RagasEvaluationRun,
    RagasEvaluationSample,
    RagasTestset,
)
from app.services.agents.graph import rag_graph
from app.services.agents.types import AgentState
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Configuration Variables
# ---------------------------------------------------------------------------
TENANT_ID = "c2a35f4c-4a7d-4084-939a-f8cadd71045d"
USER_ID = "6a759485-4654-497e-8059-c429edcbdd9e"  # a real user UUID belonging to the tenant
RUN_NAME = "eval-run-1"

# ---------------------------------------------------------------------------
# Manual / Interactive Execution Controls
# ---------------------------------------------------------------------------
INTERACTIVE_MODE = False  # Set to True to prompt before running each question
QUESTION_INDEX = None    # Set to a 1-based index (e.g., 1) to run ONLY that question
LIMIT = 2            # Set to an integer (e.g., 1) to run only the first N questions
START_INDEX = 29          # 1-based start index (e.g., start from question 5)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_inference")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline inference on testset questions."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=None,
        help="Run questions interactively with step-by-step confirmation.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only a single specific 1-based question index (e.g., --index 1).",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit the total number of questions to process (e.g., --limit 1).",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=None,
        help="1-based index to start processing from (e.g., --start 5).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the evaluation run (e.g., --run-name eval-run-1).",
    )
    return parser.parse_args()


def build_initial_state(
    query: str,
    tenant_id: str,
    user_id: str,
    session_id: Optional[str] = None,
) -> AgentState:
    """
    Construct initial_state exactly as the FastAPI route and rag_service do.
    """
    return {
        "query": query,
        "user_id": str(user_id),
        "tenant_id": str(tenant_id),
        "database_id": None,
        "document_id": None,
        "conversation_history": "",
        "model_id": None,
        "session_id": str(session_id) if session_id else str(uuid.uuid4()),
        "command_instruction": None,
        "compare_document_ids": None,
        "is_compare_mode": False,
        "is_summarize_mode": False,
        "invoke_sql": False,
        "invoke_rag": False,
        "mode": "doc_only",
        "orchestrator_reasoning": "",
        "progress_tokens": [],
        "sql_result": None,
        "rag_result": None,
        "db_user_id": None,
        "db_connection_id": None,
        "db_authorized_schema": None,
        "db_session_turns": None,
        "db_authorized_cols_by_table": None,
        "db_all_physical_cols_by_table": None,
        "db_valid_tables": None,
        "db_connection_engine": None,
        "db_connection_name": None,
        "db_is_admin": False,
        "context_error": None,
        "query_plan": None,
        "db_filtered_schema": None,
        "generated_sql": None,
        "previous_sql": None,
        "sql_generation_attempts": 0,
        "sql_generation_error": None,
        "final_answer": "",
        "citations": [],
        "follow_up_questions": [],
        "chart_spec": None,
        "query_results": None,
        "model_string": None,
        "resolved_model": None,
        "resolved_model_id": None,
        "was_fallback": False,
        "fallback_model_name": None,
        "execution_time_ms": 0,
        "answer_judge_feedback": None,
        "answer_judge_attempts": 0,
    }


def extract_contexts(state_values: dict[str, Any]) -> list[str]:
    """
    Extract retrieved context chunk texts from state values and convert them to a list of strings.
    """
    contexts: list[str] = []
    rag_result = state_values.get("rag_result")

    if rag_result:
        # Check qdrant_results in rag_result
        qdrant_results = getattr(rag_result, "qdrant_results", None)
        if qdrant_results is None and isinstance(rag_result, dict):
            qdrant_results = rag_result.get("qdrant_results")

        if qdrant_results and isinstance(qdrant_results, list):
            for hit in qdrant_results:
                payload = (
                    hit.get("payload", {})
                    if isinstance(hit, dict)
                    else getattr(hit, "payload", {})
                )
                chunk_text = (
                    payload.get("chunk_text")
                    if isinstance(payload, dict)
                    else getattr(payload, "chunk_text", None)
                )
                if chunk_text and isinstance(chunk_text, str):
                    contexts.append(chunk_text)

        # Check excel_results in rag_result
        excel_results = getattr(rag_result, "excel_results", None)
        if excel_results is None and isinstance(rag_result, dict):
            excel_results = rag_result.get("excel_results")

        if excel_results and isinstance(excel_results, list):
            for er in excel_results:
                res = (
                    er.get("result")
                    if isinstance(er, dict)
                    else getattr(er, "result", None)
                )
                if res:
                    contexts.append(str(res))

    # Fallback to citations if contexts list is still empty
    if not contexts:
        citations = state_values.get("citations") or []
        if isinstance(citations, list):
            for cit in citations:
                chunk_text = (
                    cit.get("chunk_text")
                    if isinstance(cit, dict)
                    else getattr(cit, "chunk_text", None)
                )
                if chunk_text and isinstance(chunk_text, str):
                    contexts.append(chunk_text)

    return contexts


async def run_single_inference(
    question: str,
    tenant_id: str,
    user_id: str,
) -> tuple[str, list[str]]:
    """
    Run a single question through the LangGraph RAG pipeline and extract answer + context chunks.
    """
    session_id = str(uuid.uuid4())
    initial_state = build_initial_state(
        query=question,
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
    )
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    # Stream graph execution and consume full stream
    async for _ in rag_graph.astream(initial_state, config=config):
        pass

    # Retrieve full final state from checkpointer
    full_final_state = await rag_graph.aget_state(config)
    state_values = full_final_state.values if full_final_state else {}

    final_answer = state_values.get("final_answer", "")
    contexts = extract_contexts(state_values)

    return final_answer, contexts


async def main() -> None:
    # Read CLI arguments if provided, falling back to top-level constants
    cli_args = parse_args() if "pytest" not in sys.modules else None
    
    interactive = (
        cli_args.interactive if (cli_args and cli_args.interactive is not None) else INTERACTIVE_MODE
    )
    target_index = (
        cli_args.index if (cli_args and cli_args.index is not None) else QUESTION_INDEX
    )
    limit = (
        cli_args.limit if (cli_args and cli_args.limit is not None) else LIMIT
    )
    start_idx = (
        cli_args.start if (cli_args and cli_args.start is not None) else START_INDEX
    )

    logger.info("=== Starting RAG Pipeline Inference Evaluation Run ===")
    if interactive:
        logger.info("Mode: INTERACTIVE (manual confirmation per question enabled)")
    if target_index:
        logger.info(f"Target: Running ONLY Question #{target_index}")
    elif limit:
        logger.info(f"Limit: Processing maximum {limit} question(s) starting from #{start_idx}")

    # Validate configuration parameters
    if not TENANT_ID or TENANT_ID == "your-tenant-uuid-here":
        logger.error(
            "Please set a valid TENANT_ID at the top of scripts/run_inference.py before running."
        )
        sys.exit(1)

    if not USER_ID or USER_ID == "your-user-uuid-here":
        logger.error(
            "Please set a valid USER_ID at the top of scripts/run_inference.py before running."
        )
        sys.exit(1)

    try:
        tenant_uuid = uuid.UUID(str(TENANT_ID))
    except ValueError:
        logger.error(f"Invalid UUID format for TENANT_ID: '{TENANT_ID}'")
        sys.exit(1)

    try:
        uuid.UUID(str(USER_ID))
    except ValueError:
        logger.error(f"Invalid UUID format for USER_ID: '{USER_ID}'")
        sys.exit(1)

    db: Session = SessionLocal()
    processed_count = 0
    inserted_count = 0
    skipped_count = 0

    try:
        # Load all rows from ragas_testset where tenant_id = TENANT_ID
        logger.info(f"Fetching testset rows for tenant_id: {tenant_uuid}...")
        testset_rows = (
            db.query(RagasTestset)
            .filter(RagasTestset.tenant_id == tenant_uuid)
            .order_by(RagasTestset.created_at.asc())
            .all()
        )

        total = len(testset_rows)
        logger.info(f"Found {total} testset question(s) in database.")

        if total == 0:
            logger.warning(
                f"No testset questions found in 'ragas_testset' table for tenant '{TENANT_ID}'."
            )
            return

        # Filter by target index or start/limit if requested
        if target_index is not None:
            if 1 <= target_index <= total:
                rows_to_process = [(target_index - 1, testset_rows[target_index - 1])]
            else:
                logger.error(
                    f"Invalid --index {target_index}. Must be between 1 and {total}."
                )
                return
        else:
            start_0_based = max(0, (start_idx or 1) - 1)
            selected_rows = testset_rows[start_0_based:]
            if limit:
                selected_rows = selected_rows[:limit]
            rows_to_process = [
                (start_0_based + idx, row) for idx, row in enumerate(selected_rows)
            ]

        # Fetch or create the evaluation run record for this run_name and tenant
        run_name = (
            cli_args.run_name
            if (cli_args and getattr(cli_args, "run_name", None))
            else RUN_NAME
        )
        eval_run = (
            db.query(RagasEvaluationRun)
            .filter(
                RagasEvaluationRun.tenant_id == tenant_uuid,
                RagasEvaluationRun.run_name == run_name,
            )
            .first()
        )
        if not eval_run:
            eval_run = RagasEvaluationRun(
                id=uuid.uuid4(),
                tenant_id=tenant_uuid,
                run_name=run_name,
            )
            db.add(eval_run)
            db.commit()
            db.refresh(eval_run)
            logger.info(f"Created new evaluation run '{run_name}' (ID: {eval_run.id})")
        else:
            logger.info(f"Using existing evaluation run '{run_name}' (ID: {eval_run.id})")

        for i, row in rows_to_process:
            q_num = i + 1
            print("\n" + "-" * 70)
            print(f"[{q_num}/{total}] Question: {row.question}")
            print("-" * 70)

            if interactive:
                user_choice = input(
                    f"-> Run question #{q_num}? [Enter]=Run | [s]=Skip | [q]=Quit: "
                ).strip().lower()

                if user_choice == "q":
                    print("User requested exit. Stopping run...")
                    break
                elif user_choice == "s":
                    print(f"Skipping question #{q_num}.")
                    skipped_count += 1
                    continue

            processed_count += 1
            try:
                print(f"Running inference for question #{q_num} through LangGraph RAG pipeline...")
                final_answer, contexts = await run_single_inference(
                    question=row.question,
                    tenant_id=TENANT_ID,
                    user_id=USER_ID,
                )

                # Insert row into ragas_evaluation_samples
                sample = RagasEvaluationSample(
                    id=uuid.uuid4(),
                    run_id=eval_run.id,
                    question=row.question,
                    ground_truth=row.ground_truth,
                    contexts=contexts,
                    answer=final_answer,
                    faithfulness=None,
                    answer_relevancy=None,
                    context_precision=None,
                    context_recall=None,
                )
                db.add(sample)
                db.commit()
                inserted_count += 1

                print(f"\n[Done #{q_num}] Generated Answer ({len(final_answer)} chars):")
                print(f"{final_answer}")
                print(f"\nRetrieved {len(contexts)} context chunk(s). Inserted sample into database.")

                if interactive:
                    input("\nPress [Enter] to continue to the next question...")

            except Exception as exc:
                db.rollback()
                logger.warning(
                    f"[{q_num}/{total}] Error processing question '{row.question[:60]}...': {exc}"
                )
                print(
                    f"[{q_num}/{total}] Warning: Failed to process question '{row.question[:60]}...': {exc}"
                )
                continue

    finally:
        db.close()

    print("\n" + "=" * 50)
    print("=== Inference Run Summary ===")
    print(f"Total questions processed: {processed_count}")
    print(f"Total inserted into DB:    {inserted_count}")
    if skipped_count:
        print(f"Total skipped:             {skipped_count}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
