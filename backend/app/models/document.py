from datetime import datetime
import uuid
from sqlalchemy import String, DateTime, Uuid, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base
from app.models.enums import (
    file_type_enum, FileType,
    owner_type_enum, OwnerType,
    visibility_enum, Visibility
)

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("tenants.id", ondelete="CASCADE"), 
        nullable=False
    )
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[FileType] = mapped_column(file_type_enum, nullable=False)
    owner_type: Mapped[OwnerType] = mapped_column(owner_type_enum, nullable=False)
    visibility: Mapped[Visibility] = mapped_column(visibility_enum, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    qdrant_collection: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(), 
        nullable=False
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="documents")
    uploader: Mapped["User"] = relationship("User", back_populates="documents")
    access_policies: Mapped[list["DocumentAccessPolicy"]] = relationship("DocumentAccessPolicy", back_populates="document", cascade="all, delete-orphan")
    citations: Mapped[list["QueryCitation"]] = relationship("QueryCitation", back_populates="document", cascade="all, delete-orphan")
