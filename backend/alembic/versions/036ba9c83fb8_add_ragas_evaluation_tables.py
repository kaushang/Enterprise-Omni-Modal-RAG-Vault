"""add_ragas_evaluation_tables

Revision ID: 036ba9c83fb8
Revises: 3afcec7e563c
Create Date: 2026-08-17 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "036ba9c83fb8"
down_revision: Union[str, Sequence[str], None] = "3afcec7e563c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ragas_testset",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("ground_truth", sa.Text(), nullable=False),
        sa.Column(
            "reference_contexts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_ragas_testset_tenant", "ragas_testset", ["tenant_id"], unique=False
    )

    op.create_table(
        "ragas_evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("run_name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("avg_faithfulness", sa.Float(), nullable=True),
        sa.Column("avg_answer_relevancy", sa.Float(), nullable=True),
        sa.Column("avg_context_precision", sa.Float(), nullable=True),
        sa.Column("avg_context_recall", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_ragas_eval_runs_tenant",
        "ragas_evaluation_runs",
        ["tenant_id"],
        unique=False,
    )

    op.create_table(
        "ragas_evaluation_samples",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("ground_truth", sa.Text(), nullable=False),
        sa.Column("contexts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("answer_relevancy", sa.Float(), nullable=True),
        sa.Column("context_precision", sa.Float(), nullable=True),
        sa.Column("context_recall", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["ragas_evaluation_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_ragas_eval_samples_run",
        "ragas_evaluation_samples",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_ragas_eval_samples_run", table_name="ragas_evaluation_samples")
    op.drop_table("ragas_evaluation_samples")
    op.drop_index("idx_ragas_eval_runs_tenant", table_name="ragas_evaluation_runs")
    op.drop_table("ragas_evaluation_runs")
    op.drop_index("idx_ragas_testset_tenant", table_name="ragas_testset")
    op.drop_table("ragas_testset")
