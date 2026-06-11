from datetime import datetime
import uuid
from sqlalchemy import String, Text, Integer, DateTime, Uuid, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class QueryCitation(Base):
    __tablename__ = "query_citations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("query_messages.id", ondelete="CASCADE"), 
        nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("documents.id", ondelete="CASCADE"), 
        nullable=False
    )
    qdrant_vector_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    message: Mapped["QueryMessage"] = relationship("QueryMessage", back_populates="citations")
    document: Mapped["Document"] = relationship("Document", back_populates="citations")
