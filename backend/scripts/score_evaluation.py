import argparse
import json
import logging
import math
import os
import sys
import uuid
from typing import Any

from app.db.session import SessionLocal
from app.models.ragas import (
    RagasEvaluationRun,
    RagasEvaluationSample,
)
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from sqlalchemy import or_
from sqlalchemy.orm import Session

# Ensure the backend directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env
load_dotenv(find_dotenv())


# ---------------------------------------------------------------------------
# Configuration Variables
# ---------------------------------------------------------------------------
TENANT_ID = "c2a35f4c-4a7d-4084-939a-f8cadd71045d"
RUN_NAME = "eval-run-1"
BATCH_SIZE = None  # number of unscored samples to evaluate in this run, set to None to evaluate all pending
SAMPLE_IDS = []  # optionally hardcode specific ragas_evaluation_samples UUIDs to evaluate, overrides BATCH_SIZE if non-empty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("score_evaluation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAGAS metric evaluation on unscored inference samples."
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default=None,
        help="Tenant UUID to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the evaluation run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of samples to evaluate in this batch (set to 0 or leave empty for all).",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Specific sample UUIDs to evaluate.",
    )
    return parser.parse_args()


def safe_float(val: Any) -> float | None:
    """
    Safely convert a metric output to a Python float or None.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def safe_mean(values: list[float | None]) -> float | None:
    """
    Calculate the arithmetic mean of valid, non-null float values.
    """
    valid_nums = [v for v in values if v is not None and not math.isnan(v)]
    if not valid_nums:
        return None
    return float(sum(valid_nums) / len(valid_nums))


def format_score(score: float | None) -> str:
    """
    Format a score for printing in the summary table.
    """
    return f"{score:.2f}" if score is not None else "N/A"


def build_evaluation_dataset(samples: list[RagasEvaluationSample]) -> EvaluationDataset:
    """
    Construct a RAGAS EvaluationDataset from database sample records.
    """
    eval_samples: list[SingleTurnSample] = []
    for sample in samples:
        raw_contexts = sample.contexts
        if isinstance(raw_contexts, str):
            try:
                contexts_list = json.loads(raw_contexts)
            except Exception:
                contexts_list = [raw_contexts]
        elif isinstance(raw_contexts, list):
            contexts_list = raw_contexts
        else:
            contexts_list = []

        clean_contexts = [str(c) for c in contexts_list if c is not None]
        if not clean_contexts:
            clean_contexts = [""]

        eval_samples.append(
            SingleTurnSample(
                user_input=sample.question or "",
                response=sample.answer or "",
                retrieved_contexts=clean_contexts,
                reference=sample.ground_truth or "",
            )
        )

    return EvaluationDataset(samples=eval_samples)


def load_pending_samples(
    db: Session,
    tenant_uuid: uuid.UUID,
    sample_ids: list[str] | None = None,
    batch_size: int | None = None,
) -> list[RagasEvaluationSample]:
    """
    Query unscored samples ensuring idempotency (faithfulness IS NULL).
    """
    if sample_ids:
        uuids = [uuid.UUID(str(sid)) for sid in sample_ids]
        return (
            db.query(RagasEvaluationSample)
            .outerjoin(
                RagasEvaluationRun,
                RagasEvaluationSample.run_id == RagasEvaluationRun.id,
            )
            .filter(
                or_(
                    RagasEvaluationSample.tenant_id == tenant_uuid,
                    RagasEvaluationRun.tenant_id == tenant_uuid,
                ),
                RagasEvaluationSample.id.in_(uuids),
                RagasEvaluationSample.faithfulness.is_(None),
            )
            .order_by(RagasEvaluationSample.id.asc())
            .all()
        )

    query = (
        db.query(RagasEvaluationSample)
        .outerjoin(
            RagasEvaluationRun, RagasEvaluationSample.run_id == RagasEvaluationRun.id
        )
        .filter(
            or_(
                RagasEvaluationSample.tenant_id == tenant_uuid,
                RagasEvaluationRun.tenant_id == tenant_uuid,
            ),
            RagasEvaluationSample.faithfulness.is_(None),
        )
        .order_by(RagasEvaluationSample.id.asc())
    )

    if batch_size is not None and batch_size > 0:
        query = query.limit(batch_size)

    return query.all()


def main() -> None:
    cli_args = parse_args() if "pytest" not in sys.modules else None

    tenant_id_str = (
        cli_args.tenant_id if (cli_args and cli_args.tenant_id) else TENANT_ID
    )
    run_name = cli_args.run_name if (cli_args and cli_args.run_name) else RUN_NAME
    sample_ids = (
        cli_args.sample_ids
        if (cli_args and cli_args.sample_ids is not None)
        else SAMPLE_IDS
    )
    batch_size = (
        cli_args.batch_size
        if (cli_args and cli_args.batch_size is not None)
        else BATCH_SIZE
    )

    # Validate TENANT_ID
    if not tenant_id_str or tenant_id_str == "your-tenant-uuid-here":
        logger.error(
            "Please set a valid TENANT_ID at the top of scripts/score_evaluation.py before running."
        )
        sys.exit(1)

    try:
        tenant_uuid = uuid.UUID(str(tenant_id_str))
    except ValueError:
        logger.error(f"Invalid UUID format for TENANT_ID: '{tenant_id_str}'")
        sys.exit(1)

    db: Session = SessionLocal()

    try:
        # Load unscored samples
        samples = load_pending_samples(
            db=db,
            tenant_uuid=tenant_uuid,
            sample_ids=sample_ids,
            batch_size=batch_size,
        )

        if not samples:
            print("No pending samples found for evaluation.")
            return

        print(f"Loaded {len(samples)} pending sample(s) for evaluation.")

        # Configure the RAGAS evaluator LLM using Claude Haiku
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY environment variable is not set.")

        evaluator_llm = LangchainLLMWrapper(
            ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=anthropic_api_key,
            )
        )

        # Configure the RAGAS embedding model using GoogleGenerativeAIEmbeddings
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            logger.warning(
                "GOOGLE_API_KEY / GEMINI_API_KEY environment variable is not set."
            )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=google_api_key,
            )
        )

        # Initialize RAGAS metrics with the evaluator LLM and embeddings
        faithfulness.llm = evaluator_llm
        answer_relevancy.llm = evaluator_llm
        answer_relevancy.embeddings = evaluator_embeddings
        context_precision.llm = evaluator_llm
        context_recall.llm = evaluator_llm

        # Build RAGAS EvaluationDataset
        evaluation_dataset = build_evaluation_dataset(samples)

        # Run RAGAS evaluate()
        logger.info(f"Running RAGAS evaluation on {len(samples)} sample(s)...")
        results = evaluate(
            dataset=evaluation_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        # Convert results to a pandas DataFrame
        df = results.to_pandas()

        # Extract metric scores per row
        faithfulness_scores: list[float | None] = [
            safe_float(df["faithfulness"].iloc[i])
            if "faithfulness" in df.columns
            else None
            for i in range(len(samples))
        ]
        answer_relevancy_scores: list[float | None] = [
            safe_float(df["answer_relevancy"].iloc[i])
            if "answer_relevancy" in df.columns
            else None
            for i in range(len(samples))
        ]
        context_precision_scores: list[float | None] = [
            safe_float(df["context_precision"].iloc[i])
            if "context_precision" in df.columns
            else None
            for i in range(len(samples))
        ]
        context_recall_scores: list[float | None] = [
            safe_float(df["context_recall"].iloc[i])
            if "context_recall" in df.columns
            else None
            for i in range(len(samples))
        ]

        # Calculate mean scores for the batch
        avg_faithfulness = safe_mean(faithfulness_scores)
        avg_answer_relevancy = safe_mean(answer_relevancy_scores)
        avg_context_precision = safe_mean(context_precision_scores)
        avg_context_recall = safe_mean(context_recall_scores)

        # Create new RagasEvaluationRun record
        new_eval_run = RagasEvaluationRun(
            id=uuid.uuid4(),
            tenant_id=tenant_uuid,
            run_name=run_name,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_precision=avg_context_precision,
            avg_context_recall=avg_context_recall,
        )
        db.add(new_eval_run)
        db.flush()

        # Update each scored sample row
        for idx, sample in enumerate(samples):
            sample.run_id = new_eval_run.id
            sample.tenant_id = tenant_uuid
            sample.faithfulness = faithfulness_scores[idx]
            sample.answer_relevancy = answer_relevancy_scores[idx]
            sample.context_precision = context_precision_scores[idx]
            sample.context_recall = context_recall_scores[idx]

        db.commit()

        # Print summary table
        print("\n========== RAGAS Evaluation Summary ==========")
        print(f"Run Name         : {run_name}")
        print(f"Samples Evaluated: {len(samples)}")
        print(f"Avg Faithfulness      : {format_score(avg_faithfulness)}")
        print(f"Avg Answer Relevancy  : {format_score(avg_answer_relevancy)}")
        print(f"Avg Context Precision : {format_score(avg_context_precision)}")
        print(f"Avg Context Recall    : {format_score(avg_context_recall)}")
        print("===============================================")

    except Exception as exc:
        db.rollback()
        logger.error(f"RAGAS evaluation failed: {exc}", exc_info=True)
        print(f"Error during RAGAS evaluation: {exc}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
