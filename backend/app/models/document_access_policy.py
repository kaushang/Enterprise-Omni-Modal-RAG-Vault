from datetime import datetime
import uuid
from sqlalchemy import DateTime, Uuid, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base
from app.models.enums import user_role_enum, UserRole

class DocumentAccessPolicy(Base):
    __tablename__ = "document_access_policies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False
    )
    role: Mapped[UserRole] = mapped_column(user_role_enum, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Unique Constraint
    __table_args__ = (
        UniqueConstraint("document_id", "role", name="uq_document_id_role"),
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="access_policies")
