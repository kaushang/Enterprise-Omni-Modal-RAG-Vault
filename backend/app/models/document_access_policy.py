from datetime import datetime
import uuid
from sqlalchemy import DateTime, Uuid, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.role import Role
    
class DocumentAccessPolicy(Base):
    __tablename__ = "document_access_policies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("roles.id", ondelete="CASCADE"), 
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Unique Constraint
    __table_args__ = (
        UniqueConstraint("document_id", "role_id", name="uq_document_id_role_id"),
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="access_policies")
    role: Mapped["Role"] = relationship("Role")
