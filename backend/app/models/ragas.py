import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.tenant import Tenant


class RagasTestset(Base):
    __tablename__ = "ragas_testset"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    reference_contexts: Mapped[List[str]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_ragas_testset_tenant", "tenant_id"),)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant")


class RagasEvaluationRun(Base):
    __tablename__ = "ragas_evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    avg_faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_answer_relevancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_context_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_context_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("idx_ragas_eval_runs_tenant", "tenant_id"),)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant")
    samples: Mapped[List["RagasEvaluationSample"]] = relationship(
        "RagasEvaluationSample",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class RagasEvaluationSample(Base):
    __tablename__ = "ragas_evaluation_samples"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("ragas_evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False)
    contexts: Mapped[List[str]] = mapped_column(JSONB, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("idx_ragas_eval_samples_run", "run_id"),)

    # Relationships
    run: Mapped["RagasEvaluationRun"] = relationship(
        "RagasEvaluationRun", back_populates="samples"
    )


# Aliases for flexibility
RAGASTestset = RagasTestset
RAGASEvaluationRun = RagasEvaluationRun
RAGASEvaluationSample = RagasEvaluationSample
