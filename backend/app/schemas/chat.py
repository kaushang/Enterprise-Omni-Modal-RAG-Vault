"""
Pydantic schemas for the Chat / RAG query endpoints.
"""

import json
from pydantic import BaseModel, Field, model_validator, field_validator
from uuid import UUID
from datetime import datetime
from typing import Optional, Any


class ModelResponse(BaseModel):
    id: UUID
    display_name: str
    is_active: bool
    created_at: datetime
    provider_id: Optional[str] = None
    base_url: Optional[str] = None
    input_cost_per_million_tokens: Optional[float] = None
    output_cost_per_million_tokens: Optional[float] = None
    tenant_id: Optional[UUID] = None
    api_key: Optional[str] = ""
    is_default: bool = False
    model_name: Optional[str] = None
    # True when this model is the tenant's admin-configured default chat model
    is_tenant_default: bool = False
    tier: str = "balanced"

    model_config = {"from_attributes": True}


class SessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    tenant_id: UUID
    title: Optional[str] = None
    is_pinned: bool = False
    db_connection_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CreateSessionResponse(SessionResponse):
    """Alias used for session creation responses (identical fields)."""

    pass


class CitationResponse(BaseModel):
    id: UUID
    document_id: Optional[UUID] = None
    connection_id: Optional[UUID] = None
    filename: str = ""
    chunk_text: str
    page_number: Optional[int] = None
    chunk_index: int

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def resolve_filename(cls, data: Any) -> Any:
        """
        When building from a SQLAlchemy ORM QueryCitation object,
        extract the filename from the related document or database relationship.
        """
        if hasattr(data, "document") and data.document is not None:
            # Inject filename from the related Document
            data.__dict__["filename"] = data.document.filename
        elif hasattr(data, "connection") and data.connection is not None:
            # Inject connection name
            data.__dict__["filename"] = f"Database: {data.connection.name}"
        return data


class MessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    created_at: datetime
    citations: list[CitationResponse] = []
    attached_file: Optional[dict] = None
    model_id: Optional[UUID] = None
    model: Optional[ModelResponse] = None
    follow_up_questions: Optional[list[str]] = None
    generated_sql: Optional[str] = None
    query_results: Optional[list[dict]] = None
    chart_spec: Optional[dict] = None
    resolved_model: Optional[str] = None
    was_fallback: bool = False
    fallback_model_name: Optional[str] = None

    model_config = {"from_attributes": True}

    @field_validator("query_results", mode="before")
    @classmethod
    def validate_query_results(cls, v: Any) -> Optional[list[dict[str, Any]]]:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                return None

        if isinstance(v, dict):
            return [v]

        if isinstance(v, list):
            valid_rows = [item for item in v if isinstance(item, dict)]
            return valid_rows if valid_rows or len(v) == 0 else None

        return None

    @field_validator("follow_up_questions", mode="before")
    @classmethod
    def validate_follow_up_questions(cls, v: Any) -> Optional[list[str]]:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                return None

        if isinstance(v, list):
            return [str(item) for item in v]

        return None

    @field_validator("chart_spec", mode="before")
    @classmethod
    def validate_chart_spec(cls, v: Any) -> Optional[dict[str, Any]]:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                return None

        if isinstance(v, dict):
            return v

        return None

    @model_validator(mode="before")
    @classmethod
    def resolve_attached_file(cls, data: Any) -> Any:
        if isinstance(data, dict):
            attached_doc = data.get("attached_document")
            if attached_doc:
                if hasattr(attached_doc, "filename"):
                    data["attached_file"] = {
                        "name": attached_doc.filename,
                        "size": getattr(attached_doc, "file_size", 0) or 0,
                    }
                elif isinstance(attached_doc, dict):
                    data["attached_file"] = {
                        "name": attached_doc.get("filename"),
                        "size": attached_doc.get("file_size") or 0,
                    }
        else:
            attached_doc = getattr(data, "attached_document", None)
            if attached_doc is not None:
                data.__dict__["attached_file"] = {
                    "name": attached_doc.filename,
                    "size": attached_doc.file_size or 0,
                }
        return data


class SessionDetailResponse(SessionResponse):
    messages: list[MessageResponse] = []


class QueryRequest(BaseModel):
    content: str = Field(..., min_length=1, strip_whitespace=True)
    document_id: Optional[UUID] = None
    database_id: Optional[UUID] = None
    model_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    message_id: str


class TranscriptionResponse(BaseModel):
    text: str


class MatchingLineItem(BaseModel):
    text: str
    role: str


class ChatSearchItem(BaseModel):
    conversation_id: UUID
    conversation_title: Optional[str] = None
    conversation_updated_at: datetime
    conversation_date: datetime
    matching_lines: list[MatchingLineItem]
    match_in_title: bool
    match_count: int

    model_config = {"from_attributes": True}


class ChatSearchResponse(BaseModel):
    results: list[ChatSearchItem]
    total_count: int
    has_more: bool
