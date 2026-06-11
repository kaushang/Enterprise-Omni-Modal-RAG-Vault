from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
import secrets
import hashlib
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hashes a plain password using bcrypt via passlib."""
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """Verifies a plain password against a hash."""
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    """Creates a JWT access token with expiry from ACCESS_TOKEN_EXPIRE_MINUTES in config."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    # Ensure subject, tenant_id, and role are strings
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    if "tenant_id" in to_encode:
        to_encode["tenant_id"] = str(to_encode["tenant_id"])
    if "role" in to_encode:
        to_encode["role"] = str(to_encode["role"])
        
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token() -> tuple[str, str]:
    """Generates a cryptographically secure random token and returns (raw_token, hashed_token)."""
    raw_token = secrets.token_urlsafe(64)
    hashed_token = hash_token(raw_token)
    return raw_token, hashed_token

def decode_access_token(token: str) -> dict | None:
    """Decodes and validates a JWT access token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None

def hash_token(token: str) -> str:
    """Returns SHA-256 hash of a raw token string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def generate_otp() -> tuple[str, str]:
    """Generates a cryptographically secure 6-digit numeric OTP and its SHA-256 hash."""
    otp_int = secrets.randbelow(1000000)
    raw_otp = str(otp_int).zfill(6)
    otp_hash = hashlib.sha256(raw_otp.encode("utf-8")).hexdigest()
    return raw_otp, otp_hash

def verify_otp(raw_otp: str, otp_hash: str) -> bool:
    """Hashes the raw OTP using SHA-256 and compares it to the stored hash."""
    if not raw_otp:
        return False
    computed_hash = hashlib.sha256(raw_otp.encode("utf-8")).hexdigest()
    return secrets.compare_digest(computed_hash, otp_hash)
