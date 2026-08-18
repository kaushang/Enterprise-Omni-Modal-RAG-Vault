import json
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.base import Base
from app.models.ragas import (
    RagasEvaluationRun,
    RagasEvaluationSample,
)
from app.models.tenant import Tenant
from scripts.score_evaluation import (
    build_evaluation_dataset,
    format_score,
    load_pending_samples,
    safe_float,
    safe_mean,
)

DATABASE_URL = "sqlite:///:memory:"


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"


@pytest.fixture(name="db_session_factory")
def db_session_factory_fixture():
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    yield TestingSessionLocal
    engine.dispose()


@pytest.fixture(name="db")
def db_fixture(db_session_factory):
    session = db_session_factory()
    try:
        yield session
    finally:
        session.close()


def test_safe_float_and_mean():
    assert safe_float(0.85) == 0.85
    assert safe_float("0.92") == 0.92
    assert safe_float(None) is None
    assert safe_float(float("nan")) is None
    assert safe_float("invalid") is None

    assert safe_mean([0.8, 0.9, None, 1.0]) == pytest.approx(0.9)
    assert safe_mean([None, float("nan")]) is None
    assert safe_mean([]) is None


def test_format_score():
    assert format_score(0.876) == "0.88"
    assert format_score(None) == "N/A"


def test_build_evaluation_dataset():
    sample1 = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        question="What is RAG?",
        ground_truth="Retrieval-Augmented Generation.",
        contexts=["Context chunk 1", "Context chunk 2"],
        answer="RAG is Retrieval-Augmented Generation.",
    )
    sample2 = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        question="What is SQL?",
        ground_truth="Structured Query Language.",
        contexts=json.dumps(["SQL chunk"]),
        answer="SQL is Structured Query Language.",
    )

    dataset = build_evaluation_dataset([sample1, sample2])
    assert len(dataset) == 2

    df = dataset.to_pandas()
    assert df["user_input"].iloc[0] == "What is RAG?"
    assert df["response"].iloc[0] == "RAG is Retrieval-Augmented Generation."
    assert df["retrieved_contexts"].iloc[0] == ["Context chunk 1", "Context chunk 2"]
    assert df["reference"].iloc[0] == "Retrieval-Augmented Generation."

    assert df["user_input"].iloc[1] == "What is SQL?"
    assert df["retrieved_contexts"].iloc[1] == ["SQL chunk"]


def test_load_pending_samples_idempotency_and_batching(db):
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    db.add(tenant)
    db.commit()

    initial_run = RagasEvaluationRun(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        run_name="initial-run",
    )
    db.add(initial_run)
    db.commit()

    # Create 3 unscored samples and 1 scored sample
    sample_unscored_1 = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=initial_run.id,
        question="Q1",
        ground_truth="GT1",
        contexts=["C1"],
        answer="A1",
        faithfulness=None,
    )
    sample_unscored_2 = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=initial_run.id,
        question="Q2",
        ground_truth="GT2",
        contexts=["C2"],
        answer="A2",
        faithfulness=None,
    )
    sample_unscored_3 = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=initial_run.id,
        question="Q3",
        ground_truth="GT3",
        contexts=["C3"],
        answer="A3",
        faithfulness=None,
    )
    sample_already_scored = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=initial_run.id,
        question="Q4",
        ground_truth="GT4",
        contexts=["C4"],
        answer="A4",
        faithfulness=0.95,
        answer_relevancy=0.90,
    )

    db.add_all(
        [sample_unscored_1, sample_unscored_2, sample_unscored_3, sample_already_scored]
    )
    db.commit()

    # Test loading all pending
    pending = load_pending_samples(db, tenant.id, batch_size=None)
    assert len(pending) == 3
    assert all(s.faithfulness is None for s in pending)

    # Test batch size limit
    pending_batch = load_pending_samples(db, tenant.id, batch_size=2)
    assert len(pending_batch) == 2

    # Test loading specific sample IDs
    pending_specific = load_pending_samples(
        db,
        tenant.id,
        sample_ids=[str(sample_unscored_2.id), str(sample_already_scored.id)],
    )
    # Already scored sample must not be loaded
    assert len(pending_specific) == 1
    assert pending_specific[0].id == sample_unscored_2.id


def test_full_scoring_flow_and_run_creation(db_session_factory):
    from scripts.score_evaluation import main

    db = db_session_factory()
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    db.add(tenant)
    db.commit()

    initial_run = RagasEvaluationRun(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        run_name="eval-initial",
    )
    db.add(initial_run)
    db.commit()

    sample = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=initial_run.id,
        question="What is LangGraph?",
        ground_truth="A library for stateful multi-agent workflows.",
        contexts=["LangGraph context."],
        answer="LangGraph builds stateful multi-agent workflows.",
        faithfulness=None,
    )
    db.add(sample)
    db.commit()

    tenant_id_str = str(tenant.id)
    sample_id = sample.id
    db.close()

    mock_df = pd.DataFrame(
        [
            {
                "user_input": "What is LangGraph?",
                "response": "LangGraph builds stateful multi-agent workflows.",
                "retrieved_contexts": ["LangGraph context."],
                "reference": "A library for stateful multi-agent workflows.",
                "faithfulness": 0.85,
                "answer_relevancy": 0.90,
                "context_precision": 0.80,
                "context_recall": 0.75,
            }
        ]
    )

    mock_result = MagicMock()
    mock_result.to_pandas.return_value = mock_df

    with (
        patch("scripts.score_evaluation.TENANT_ID", tenant_id_str),
        patch("scripts.score_evaluation.RUN_NAME", "eval-run-test"),
        patch("scripts.score_evaluation.SessionLocal", side_effect=db_session_factory),
        patch("scripts.score_evaluation.evaluate", return_value=mock_result),
        patch("scripts.score_evaluation.ChatAnthropic"),
        patch("scripts.score_evaluation.GoogleGenerativeAIEmbeddings"),
    ):
        main()

    # Verify run created and sample updated using a fresh session
    verify_db = db_session_factory()
    try:
        eval_run = (
            verify_db.query(RagasEvaluationRun)
            .filter(RagasEvaluationRun.run_name == "eval-run-test")
            .first()
        )
        assert eval_run is not None
        assert str(eval_run.tenant_id) == tenant_id_str
        assert eval_run.avg_faithfulness == pytest.approx(0.85)
        assert eval_run.avg_answer_relevancy == pytest.approx(0.90)
        assert eval_run.avg_context_precision == pytest.approx(0.80)
        assert eval_run.avg_context_recall == pytest.approx(0.75)

        updated_sample = (
            verify_db.query(RagasEvaluationSample)
            .filter(RagasEvaluationSample.id == sample_id)
            .first()
        )
        assert updated_sample.run_id == eval_run.id
        assert updated_sample.faithfulness == pytest.approx(0.85)
        assert updated_sample.answer_relevancy == pytest.approx(0.90)
        assert updated_sample.context_precision == pytest.approx(0.80)
        assert updated_sample.context_recall == pytest.approx(0.75)
    finally:
        verify_db.close()


def test_scoring_error_handles_rollback(db_session_factory):
    from scripts.score_evaluation import main

    db = db_session_factory()
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    db.add(tenant)
    db.commit()

    initial_run = RagasEvaluationRun(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        run_name="eval-initial",
    )
    db.add(initial_run)
    db.commit()

    sample = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=initial_run.id,
        question="What is an Agent?",
        ground_truth="An autonomous LLM system.",
        contexts=["Context."],
        answer="An Agent is an autonomous system.",
        faithfulness=None,
    )
    db.add(sample)
    db.commit()

    tenant_id_str = str(tenant.id)
    initial_run_id = initial_run.id
    sample_id = sample.id
    db.close()

    with (
        patch("scripts.score_evaluation.TENANT_ID", tenant_id_str),
        patch("scripts.score_evaluation.RUN_NAME", "eval-run-failed"),
        patch("scripts.score_evaluation.SessionLocal", side_effect=db_session_factory),
        patch(
            "scripts.score_evaluation.evaluate",
            side_effect=RuntimeError("API quota exceeded"),
        ),
        patch("scripts.score_evaluation.ChatAnthropic"),
        patch("scripts.score_evaluation.GoogleGenerativeAIEmbeddings"),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1

    # Assert no new run was created and sample remains unscored
    verify_db = db_session_factory()
    try:
        failed_run = (
            verify_db.query(RagasEvaluationRun)
            .filter(RagasEvaluationRun.run_name == "eval-run-failed")
            .first()
        )
        assert failed_run is None

        unmodified_sample = (
            verify_db.query(RagasEvaluationSample)
            .filter(RagasEvaluationSample.id == sample_id)
            .first()
        )
        assert unmodified_sample.faithfulness is None
        assert unmodified_sample.run_id == initial_run_id
    finally:
        verify_db.close()


def test_samples_persist_when_evaluation_run_is_deleted(db):
    tenant = Tenant(id=uuid.uuid4(), name="Test Tenant", slug="test-tenant")
    db.add(tenant)
    db.commit()

    run = RagasEvaluationRun(
        id=uuid.uuid4(),
        tenant_id=tenant.id,
        run_name="temporary-run",
    )
    db.add(run)
    db.commit()

    sample = RagasEvaluationSample(
        id=uuid.uuid4(),
        run_id=run.id,
        tenant_id=tenant.id,
        question="Sample question?",
        ground_truth="Sample GT",
        contexts=["Sample context"],
        answer="Sample answer",
    )
    db.add(sample)
    db.commit()

    # Delete the evaluation run
    db.delete(run)
    db.commit()

    # Verify that the sample STILL exists and is not deleted
    persisted_sample = (
        db.query(RagasEvaluationSample)
        .filter(RagasEvaluationSample.id == sample.id)
        .first()
    )
    assert persisted_sample is not None
    assert persisted_sample.run_id is None
    assert persisted_sample.tenant_id == tenant.id
    assert persisted_sample.question == "Sample question?"
