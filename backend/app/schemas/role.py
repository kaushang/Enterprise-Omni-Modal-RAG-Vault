from pydantic import BaseModel, Field
from app.schemas.auth import RoleResponse

class CreateRoleRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)

class UpdateRoleRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)

__all__ = ["CreateRoleRequest", "UpdateRoleRequest", "RoleResponse"]
