import enum
from sqlalchemy import Enum as SQLEnum

class UserRole(str, enum.Enum):
    admin = "admin"
    member = "member"

class FileType(str, enum.Enum):
    text = "text"
    audio = "audio"
    pdf = "pdf"
    docx = "docx"
    pptx = "pptx"
    excel = "excel"

class OwnerType(str, enum.Enum):
    organisation = "organisation"
    private = "private"

class Visibility(str, enum.Enum):
    org_wide = "org_wide"
    private = "private"

class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"

class OTPPurpose(str, enum.Enum):
    registration = "registration"
    forgot_password = "forgot_password"

# SQLAlchemy Enum Types
user_role_enum = SQLEnum(UserRole, name="userrole")
file_type_enum = SQLEnum(FileType, name="filetype")
owner_type_enum = SQLEnum(OwnerType, name="ownertype")
visibility_enum = SQLEnum(Visibility, name="visibility")
message_role_enum = SQLEnum(MessageRole, name="messagerole")
otp_purpose_enum = SQLEnum(OTPPurpose, name="otppurpose")
