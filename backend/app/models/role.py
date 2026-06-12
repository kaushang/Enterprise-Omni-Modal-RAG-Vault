from datetime import datetime
import uuid
from sqlalchemy import String, DateTime, Uuid, ForeignKey, Boolean, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.tenant import Tenant
    from app.models.user import User

class Role(Base):
    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("tenants.id", ondelete="CASCADE"), 
        nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(), 
        nullable=False
    )

    # Unique Constraint on (tenant_id, name)
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_tenant_id_name"),
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="roles")
    users: Mapped[list["User"]] = relationship("User", back_populates="role")
