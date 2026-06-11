from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Import all models here so that Alembic can detect them via Base.metadata
from app.models.tenant import Tenant
from app.models.user import User
from app.models.document import Document
from app.models.document_access_policy import DocumentAccessPolicy
from app.models.query_session import QuerySession
from app.models.query_message import QueryMessage
from app.models.query_citation import QueryCitation
from app.models.refresh_token import RefreshToken