"""refactor to admin defined roles

Revision ID: b5efd432b4c5
Revises: 6d165f45a301
Create Date: 2026-06-12 17:13:29.000780

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b5efd432b4c5'
down_revision: Union[str, Sequence[str], None] = '6d165f45a301'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema with data migration."""
    # 1. Create roles table
    op.create_table('roles',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('tenant_id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('is_admin', sa.Boolean(), nullable=False),
    sa.Column('is_default', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('tenant_id', 'name', name='uq_tenant_id_name')
    )

    # 2. Add nullable role_id columns
    op.add_column('document_access_policies', sa.Column('role_id', sa.Uuid(), nullable=True))
    op.add_column('users', sa.Column('role_id', sa.Uuid(), nullable=True))

    # 3. Data Migration: Seed roles for existing tenants and map users/policies
    bind = op.get_bind()
    tenants = bind.execute(sa.text("SELECT id FROM tenants")).fetchall()

    import uuid
    for tenant in tenants:
        tenant_id = tenant[0]

        # Create Admin default role for this tenant
        admin_role_id = uuid.uuid4()
        bind.execute(
            sa.text(
                "INSERT INTO roles (id, tenant_id, name, is_admin, is_default, created_at, updated_at) "
                "VALUES (:id, :tenant_id, 'Admin', true, true, now(), now())"
            ),
            {"id": admin_role_id, "tenant_id": tenant_id}
        )

        # Create Member default role for this tenant
        member_role_id = uuid.uuid4()
        bind.execute(
            sa.text(
                "INSERT INTO roles (id, tenant_id, name, is_admin, is_default, created_at, updated_at) "
                "VALUES (:id, :tenant_id, 'Member', false, true, now(), now())"
            ),
            {"id": member_role_id, "tenant_id": tenant_id}
        )

        # Map admin users
        bind.execute(
            sa.text("UPDATE users SET role_id = :role_id WHERE tenant_id = :tenant_id AND role = 'admin'"),
            {"role_id": admin_role_id, "tenant_id": tenant_id}
        )

        # Map member users
        bind.execute(
            sa.text("UPDATE users SET role_id = :role_id WHERE tenant_id = :tenant_id AND role = 'member'"),
            {"role_id": member_role_id, "tenant_id": tenant_id}
        )

        # Map document access policies
        bind.execute(
            sa.text(
                "UPDATE document_access_policies "
                "SET role_id = :admin_role_id "
                "FROM documents "
                "WHERE document_access_policies.document_id = documents.id "
                "AND documents.tenant_id = :tenant_id "
                "AND document_access_policies.role = 'admin'"
            ),
            {"admin_role_id": admin_role_id, "tenant_id": tenant_id}
        )

        bind.execute(
            sa.text(
                "UPDATE document_access_policies "
                "SET role_id = :member_role_id "
                "FROM documents "
                "WHERE document_access_policies.document_id = documents.id "
                "AND documents.tenant_id = :tenant_id "
                "AND document_access_policies.role = 'member'"
            ),
            {"member_role_id": member_role_id, "tenant_id": tenant_id}
        )

    # 4. Set role_id to be non-nullable now that all rows are mapped
    op.alter_column('users', 'role_id', nullable=False)
    op.alter_column('document_access_policies', 'role_id', nullable=False)

    # 5. Apply foreign keys and constraints, and clean up old role columns
    op.create_foreign_key(None, 'users', 'roles', ['role_id'], ['id'], ondelete='RESTRICT')
    op.drop_column('users', 'role')

    op.drop_constraint('uq_document_id_role', 'document_access_policies', type_='unique')
    op.create_unique_constraint('uq_document_id_role_id', 'document_access_policies', ['document_id', 'role_id'])
    op.create_foreign_key(None, 'document_access_policies', 'roles', ['role_id'], ['id'], ondelete='CASCADE')
    op.drop_column('document_access_policies', 'role')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('document_access_policies', sa.Column('role', sa.String(), nullable=True))
    op.add_column('users', sa.Column('role', sa.String(), nullable=True))

    bind = op.get_bind()
    # Recover role values from roles table
    bind.execute(
        sa.text(
            "UPDATE users SET role = CASE WHEN roles.is_admin = true THEN 'admin' ELSE 'member' END "
            "FROM roles WHERE users.role_id = roles.id"
        )
    )
    bind.execute(
        sa.text(
            "UPDATE document_access_policies SET role = CASE WHEN roles.is_admin = true THEN 'admin' ELSE 'member' END "
            "FROM roles WHERE document_access_policies.role_id = roles.id"
        )
    )

    op.alter_column('users', 'role', nullable=False)
    op.alter_column('document_access_policies', 'role', nullable=False)

    op.drop_constraint(None, 'users', type_='foreignkey')
    op.drop_column('users', 'role_id')

    op.drop_constraint(None, 'document_access_policies', type_='foreignkey')
    op.drop_constraint('uq_document_id_role_id', 'document_access_policies', type_='unique')
    op.create_unique_constraint('uq_document_id_role', 'document_access_policies', ['document_id', 'role'])
    op.drop_column('document_access_policies', 'role_id')

    op.drop_table('roles')
