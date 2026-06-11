from datetime import datetime
import uuid
from sqlalchemy import String, DateTime, Uuid, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base
from app.models.enums import otp_purpose_enum, OTPPurpose

class OTPVerification(Base):
    __tablename__ = "otp_verifications"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, nullable=False)
    otp_hash: Mapped[str] = mapped_column(String, nullable=False)
    purpose: Mapped[OTPPurpose] = mapped_column(otp_purpose_enum, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
