import sys
import os
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import Base
from app.models.tenant import Tenant
from app.models.ragas import (
    RagasTestset,
    RagasEvaluationRun,
    RagasEvaluationSample,
    RAGASTestset,
    RAGASEvaluationRun,
    RAGASEvaluationSample,
)

DATABASE_URL = "sqlite:///:memory:"


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"


@pytest.fixture(name="db")
def db_fixture():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_ragas_testset_creation_and_query(db):
    tenant = Tenant(
        id=uuid.uuid4(),
        name="Test Tenant",
        slug="test-tenant",
    )
    db.add(tenant)
    db.commit()

    testset_item = RagasTestset(
        tenant_id=tenant.id,
        question="What is enterprise RAG?",
        ground_truth="Enterprise RAG combines retrieval mechanisms with LLMs for proprietary knowledge.",
        reference_contexts=[
            "Enterprise RAG systems retrieve documents from secure corporate repositories.",
            "LLMs generate answers conditioned on the retrieved contexts.",
        ],
    )
    db.add(testset_item)
    db.commit()

    # Query back
    fetched = db.query(RagasTestset).filter_by(id=testset_item.id).first()
    assert fetched is not None
    assert fetched.tenant_id == tenant.id
    assert fetched.question == "What is enterprise RAG?"
    assert fetched.ground_truth.startswith("Enterprise RAG combines")
    assert len(fetched.reference_contexts) == 2
    assert fetched.tenant.name == "Test Tenant"
    assert fetched.created_at is not None


def test_ragas_evaluation_runs_and_samples_relations(db):
    tenant = Tenant(
        id=uuid.uuid4(),
        name="Eval Tenant",
        slug="eval-tenant",
    )
    db.add(tenant)
    db.commit()

    run = RagasEvaluationRun(
        tenant_id=tenant.id,
        run_name="Q3 RAG Benchmark",
        avg_faithfulness=0.92,
        avg_answer_relevancy=0.88,
        avg_context_precision=0.85,
        avg_context_recall=0.90,
    )
    db.add(run)
    db.commit()

    sample1 = RagasEvaluationSample(
        run_id=run.id,
        question="How does hybrid search work?",
        ground_truth="Hybrid search fuses sparse BM25 keyword matching with dense vector embeddings.",
        contexts=[
            "Vector search captures semantics.",
            "BM25 captures exact keyword matches.",
        ],
        answer="Hybrid search combines dense vectors with keyword search like BM25.",
        faithfulness=0.95,
        answer_relevancy=0.90,
        context_precision=0.88,
        context_recall=0.92,
    )
    sample2 = RagasEvaluationSample(
        run_id=run.id,
        question="What is context precision?",
        ground_truth="Context precision measures signal-to-noise ratio of retrieved chunks.",
        contexts=["Context precision evaluates retrieved chunks ranking."],
        answer="It measures if relevant chunks are ranked higher.",
        faithfulness=0.89,
        answer_relevancy=0.86,
        context_precision=0.82,
        context_recall=0.88,
    )
    db.add_all([sample1, sample2])
    db.commit()

    # Verify query and relationship
    fetched_run = db.query(RagasEvaluationRun).filter_by(id=run.id).first()
    assert fetched_run is not None
    assert fetched_run.run_name == "Q3 RAG Benchmark"
    assert fetched_run.avg_faithfulness == 0.92
    assert len(fetched_run.samples) == 2
    assert fetched_run.tenant.slug == "eval-tenant"

    # Verify back-populates from sample to run
    fetched_sample = db.query(RagasEvaluationSample).filter_by(id=sample1.id).first()
    assert fetched_sample is not None
    assert fetched_sample.run.id == run.id
    assert fetched_sample.run.run_name == "Q3 RAG Benchmark"

    # Verify cascade deletion
    db.delete(fetched_run)
    db.commit()
    remaining_samples = db.query(RagasEvaluationSample).filter_by(run_id=run.id).all()
    assert len(remaining_samples) == 0


def test_ragas_aliases():
    assert RAGASTestset is RagasTestset
    assert RAGASEvaluationRun is RagasEvaluationRun
    assert RAGASEvaluationSample is RagasEvaluationSample
