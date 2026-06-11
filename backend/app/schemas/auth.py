from pydantic import BaseModel, EmailStr, Field, field_validator
from uuid import UUID
from datetime import datetime
from app.models.enums import UserRole

class AdminRegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., min_length=8)
    organisation_name: str

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
