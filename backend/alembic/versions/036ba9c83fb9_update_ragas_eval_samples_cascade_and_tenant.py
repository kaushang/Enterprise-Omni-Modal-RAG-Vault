"""update_ragas_eval_samples_cascade_and_tenant

Revision ID: 036ba9c83fb9
Revises: 036ba9c83fb8
Create Date: 2026-08-18 10:50:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "036ba9c83fb9"
down_revision: Union[str, Sequence[str], None] = "036ba9c83fb8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Update run_id constraint from CASCADE to SET NULL and make nullable
    op.drop_constraint(
        "ragas_evaluation_samples_run_id_fkey",
        "ragas_evaluation_samples",
        type_="foreignkey",
    )
    op.alter_column(
        "ragas_evaluation_samples",
        "run_id",
        existing_type=sa.Uuid(),
        nullable=True,
    )
    op.create_foreign_key(
        "ragas_evaluation_samples_run_id_fkey",
        "ragas_evaluation_samples",
        "ragas_evaluation_runs",
        ["run_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 2. Add direct tenant_id column with CASCADE on delete of tenant
    op.add_column(
        "ragas_evaluation_samples",
        sa.Column("tenant_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "ragas_evaluation_samples_tenant_id_fkey",
        "ragas_evaluation_samples",
        "tenants",
        ["tenant_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "idx_ragas_eval_samples_tenant",
        "ragas_evaluation_samples",
        ["tenant_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "idx_ragas_eval_samples_tenant", table_name="ragas_evaluation_samples"
    )
    op.drop_constraint(
        "ragas_evaluation_samples_tenant_id_fkey",
        "ragas_evaluation_samples",
        type_="foreignkey",
    )
    op.drop_column("ragas_evaluation_samples", "tenant_id")

    op.drop_constraint(
        "ragas_evaluation_samples_run_id_fkey",
        "ragas_evaluation_samples",
        type_="foreignkey",
    )
    op.alter_column(
        "ragas_evaluation_samples",
        "run_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )
    op.create_foreign_key(
        "ragas_evaluation_samples_run_id_fkey",
        "ragas_evaluation_samples",
        "ragas_evaluation_runs",
        ["run_id"],
        ["id"],
        ondelete="CASCADE",
    )
