from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Import models here so Alembic can detect them
from app.models.otp_verification import OTPVerification
from app.models.role import Role