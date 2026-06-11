from datetime import datetime
import uuid
from sqlalchemy import String, DateTime, Uuid, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class QuerySession(Base):
    __tablename__ = "query_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, 
        ForeignKey("tenants.id", ondelete="CASCADE"), 
        nullable=False
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(), 
        nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="query_sessions")
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="query_sessions")
    messages: Mapped[list["QueryMessage"]] = relationship("QueryMessage", back_populates="session", cascade="all, delete-orphan")
