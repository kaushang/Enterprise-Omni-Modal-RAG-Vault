from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Import models here so Alembic can detect them
from app.models.tenant import Tenant
# from app.models.user import User
from app.models.role import Role
from app.models.department import Department
from app.models.document import Document
from app.models.document_access_policy import DocumentAccessPolicy
from app.models.query_session import QuerySession
from app.models.query_citation import QueryCitation
from app.models.query_message import QueryMessage
from app.models.otp_verification import OTPVerification
from app.models.notification import Notification
from app.models.available_model import AvailableModel
from app.models.usage_log import UsageLog
from app.models.query_log import QueryLog
from app.models.evaluation import EvaluationRun, EvaluationResult
from app.models.audit_log import AuditLog
from app.models.refresh_token import RefreshToken
from app.models.invite_token import InviteToken
from app.models.external_database import (
    ExternalDatabaseConnection,
    DatabaseSchemaCache,
    DatabaseAccessPolicy,
)
from app.models.db_query_log import DBQueryLog
from app.models.generated_report import GeneratedReport
from app.models.report_agent_run import ReportAgentRun
from app.models.collection import Collection
from app.models.ragas import (
    RagasTestset,
    RagasEvaluationRun,
    RagasEvaluationSample,
)
