from pydantic import BaseModel, EmailStr, Field, field_validator, HttpUrl, model_validator
from uuid import UUID
from datetime import datetime
from app.models.enums import UserRole

class RegistrationInitRequest(BaseModel):
    org_name: str = Field(..., min_length=2)
    org_website: HttpUrl
    full_name: str = Field(..., min_length=2)
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    @model_validator(mode="after")
    def validate_passwords(self) -> 'RegistrationInitRequest':
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

class VerifyRegistrationOTPRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    @model_validator(mode="after")
    def validate_passwords(self) -> 'ResetPasswordRequest':
        if self.new_password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AcceptInviteRequest(BaseModel):
    token: str
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: UserRole
    tenant_id: UUID
    is_active: bool
    created_at: datetime

    model_config = {
        "from_attributes": True
    }

class InviteMemberRequest(BaseModel):
    full_name: str
    email: EmailStr
    role: UserRole

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: UserRole) -> UserRole:
        if v != UserRole.member:
            raise ValueError("Only member role can be invited")
        return v

class MessageResponse(BaseModel):
    message: str
