from datetime import datetime, timedelta, timezone
import secrets
import re
import random
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session, joinedload
from app.db.session import get_db
from app.core.config import settings
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_token,
    generate_otp,
    verify_otp
)
from app.core.dependencies import get_current_user, require_admin
from app.models.user import User
from app.models.tenant import Tenant
from app.models.refresh_token import RefreshToken
from app.models.invite_token import InviteToken
from app.models.otp_verification import OTPVerification
from app.models.role import Role
from app.models.enums import OTPPurpose
from app.schemas.auth import (
    RegistrationInitRequest,
    VerifyRegistrationOTPRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    LoginRequest,
    AcceptInviteRequest,
    UserResponse,
    InviteMemberRequest,
    MessageResponse,
    GoogleOrgSetupRequest,
    SetPasswordRequest
)
from app.services.google_oauth_service import (
    get_google_auth_url,
    exchange_code_for_user_info,
    create_google_setup_token,
    verify_google_setup_token
)
from app.services.email_service import (
    send_invite_email,
    send_otp_email,
    send_forgot_password_otp_email
)
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

serializer = URLSafeTimedSerializer(settings.SECRET_KEY)

@router.post("/register/signup", response_model=MessageResponse)
def register_signup(
    request: RegistrationInitRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Open route to initiate registration. Generates and sends OTP, and sets temporary signed cookie."""
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists"
        )

    try:
        db.query(OTPVerification).filter(
            OTPVerification.email == request.email,
            OTPVerification.purpose == OTPPurpose.registration,
            OTPVerification.is_used == False
        ).delete()

        hashed_password = hash_password(request.password)
        raw_otp, otp_hash = generate_otp()

        cookie_payload = {
            "org_name": request.org_name,
            "org_website": str(request.org_website),
            "full_name": request.full_name,
            "email": request.email,
            "hashed_password": hashed_password
        }
        signed_cookie_value = serializer.dumps(cookie_payload)

        response.set_cookie(
            key="registration_session",
            value=signed_cookie_value,
            max_age=600,
            httponly=True,
            samesite="lax",
            secure=False
        )

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_record = OTPVerification(
            email=request.email,
            otp_hash=otp_hash,
            purpose=OTPPurpose.registration,
            expires_at=expires_at,
            is_used=False
        )
        db.add(otp_record)
        db.commit()

        send_otp_email(
            to_email=request.email,
            full_name=request.full_name,
            otp=raw_otp,
            org_name=request.org_name
        )
        logger.warning(f"Registration OTP for {request.email} is: {raw_otp}")

        return {"message": "OTP sent to your email. Please verify to complete registration."}
    except HTTPException as he:
        db.rollback()
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/register/verify-otp", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_verify_otp(
    request: VerifyRegistrationOTPRequest,
    req_obj: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """Open route to verify registration OTP and finalize user and tenant creation."""
    cookie_value = req_obj.cookies.get("registration_session")
    if not cookie_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration session expired. Please start again."
        )

    try:
        payload = serializer.loads(cookie_value, max_age=600)
    except (SignatureExpired, BadSignature):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration session expired. Please start again."
        )

    if payload.get("email") != request.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    otp_record = db.query(OTPVerification).filter(
        OTPVerification.email == request.email,
        OTPVerification.purpose == OTPPurpose.registration,
        OTPVerification.is_used == False
    ).order_by(OTPVerification.created_at.desc()).first()

    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    current_time = datetime.now(timezone.utc)
    expires_at = otp_record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at < current_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    if not verify_otp(request.otp, otp_record.otp_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    try:
        org_name = payload["org_name"]
        base_slug = re.sub(r'[^a-z0-9\-]', '', org_name.lower().replace(" ", "-"))
        slug = base_slug or "tenant"
        while db.query(Tenant).filter(Tenant.slug == slug).first():
            slug = f"{base_slug}-{random.randint(1000, 9999)}"

        tenant = Tenant(
            name=org_name,
            slug=slug
        )
        db.add(tenant)
        db.flush()

        # Create Admin default role for this tenant
        admin_role = Role(
            tenant_id=tenant.id,
            name="Admin",
            is_admin=True,
            is_default=True
        )
        db.add(admin_role)

        # Create Member default role for this tenant
        member_role = Role(
            tenant_id=tenant.id,
            name="Member",
            is_admin=False,
            is_default=True
        )
        db.add(member_role)
        db.flush()

        user = User(
            tenant_id=tenant.id,
            email=payload["email"],
            full_name=payload["full_name"],
            hashed_password=payload["hashed_password"],
            role_id=admin_role.id,
            is_active=True
        )
        db.add(user)

        db.delete(otp_record)
        db.commit()
        db.refresh(user)
        # Eager load user.role
        user = db.query(User).options(joinedload(User.role)).filter(User.id == user.id).first()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    access_token = create_access_token({
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "role_id": str(user.role_id),
        "is_admin": user.role.is_admin
    })
    raw_refresh_token, hashed_refresh_token = create_refresh_token()

    refresh_expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db_refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=hashed_refresh_token,
        expires_at=refresh_expires_at,
        is_revoked=False
    )
    db.add(db_refresh_token)
    db.commit()

    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )
    response.set_cookie(
        key="refresh_token",
        value=raw_refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )

    response.delete_cookie("registration_session")
    return user

@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password_route(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db)
):
    """Open route to request a password reset OTP."""
    generic_response = {"message": "If an account with this email exists, an OTP has been sent."}
    
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        return generic_response
        
    if not user.is_active:
        return generic_response
        
    try:
        db.query(OTPVerification).filter(
            OTPVerification.email == request.email,
            OTPVerification.purpose == OTPPurpose.forgot_password,
            OTPVerification.is_used == False
        ).delete()
        
        raw_otp, otp_hash = generate_otp()
        
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_record = OTPVerification(
            email=request.email,
            otp_hash=otp_hash,
            purpose=OTPPurpose.forgot_password,
            expires_at=expires_at,
            is_used=False
        )
        db.add(otp_record)
        db.commit()
        
        send_forgot_password_otp_email(
            to_email=request.email,
            full_name=user.full_name,
            otp=raw_otp
        )
        logger.warning(f"Forgot password OTP for {request.email} is: {raw_otp}")
    except HTTPException as he:
        db.rollback()
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )
        
    return generic_response

@router.post("/reset-password", response_model=MessageResponse)
def reset_password_route(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """Open route to reset password using verification OTP."""
    otp_record = db.query(OTPVerification).filter(
        OTPVerification.email == request.email,
        OTPVerification.purpose == OTPPurpose.forgot_password,
        OTPVerification.is_used == False
    ).order_by(OTPVerification.created_at.desc()).first()
    
    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
        
    current_time = datetime.now(timezone.utc)
    expires_at = otp_record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
        
    if expires_at < current_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
        
    if not verify_otp(request.otp, otp_record.otp_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
        
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
        
    try:
        user.hashed_password = hash_password(request.new_password)
        db.delete(otp_record)
        
        db.query(RefreshToken).filter(
            RefreshToken.user_id == user.id
        ).update({RefreshToken.is_revoked: True})
        
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    return {"message": "Password reset successfully. Please log in with your new password."}

@router.post("/login", response_model=UserResponse)
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """Open route to log in, setting HTTP-only cookies on success."""
    user = db.query(User).options(joinedload(User.role)).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active"
        )
        
    # Generate tokens
    access_token = create_access_token({
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "role_id": str(user.role_id),
        "is_admin": user.role.is_admin
    })
    
    raw_refresh_token, hashed_refresh_token = create_refresh_token()
    
    # Store hashed refresh token in database
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db_refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=hashed_refresh_token,
        expires_at=expires_at,
        is_revoked=False
    )
    db.add(db_refresh_token)
    db.commit()
    
    # Set cookies on the response
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )
    
    response.set_cookie(
        key="refresh_token",
        value=raw_refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )
    
    return user

@router.post("/refresh")
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    """Open route to obtain a new access token when the old one is expired."""
    raw_refresh_token = request.cookies.get("refresh_token")
    if not raw_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token cookie missing"
        )
        
    hashed_token = hash_token(raw_refresh_token)
    db_token = db.query(RefreshToken).filter(RefreshToken.token_hash == hashed_token).first()
    
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    if db_token.is_revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is revoked"
        )
        
    current_time = datetime.now(timezone.utc)
    expires_at = db_token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
        
    if expires_at < current_time:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is expired"
        )
        
    user = db.query(User).options(joinedload(User.role)).filter(User.id == db_token.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
        
    # Generate new access token
    access_token = create_access_token({
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "role_id": str(user.role_id),
        "is_admin": user.role.is_admin
    })
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )
    
    return {"message": "Token refreshed"}

@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected route to log out by revoking the refresh token and clearing cookies."""
    raw_refresh_token = request.cookies.get("refresh_token")
    if raw_refresh_token:
        hashed_token = hash_token(raw_refresh_token)
        db_token = db.query(RefreshToken).filter(
            RefreshToken.token_hash == hashed_token,
            RefreshToken.user_id == current_user.id
        ).first()
        if db_token:
            db_token.is_revoked = True
            db.commit()
            
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "Logged out successfully"}

@router.post("/invite-member", response_model=MessageResponse)
def invite_member(
    request: InviteMemberRequest,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Protected route (admin only) to invite a new member to their tenant."""
    # Verify the role_id exists and belongs to the admin's tenant
    role = db.query(Role).filter(Role.id == request.role_id, Role.tenant_id == current_admin.tenant_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role"
        )

    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
        
    try:
        user = User(
            tenant_id=current_admin.tenant_id,
            email=request.email,
            full_name=request.full_name,
            hashed_password="",  # Empty string for now
            role_id=role.id,
            is_active=False
        )
        db.add(user)
        db.flush()  # Obtain user.id
        
        # Create invite token
        raw_token = secrets.token_urlsafe(64)
        hashed_token = hash_token(raw_token)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=48)
        
        invite_token = InviteToken(
            user_id=user.id,
            token_hash=hashed_token,
            expires_at=expires_at,
            is_used=False
        )
        db.add(invite_token)
        db.commit()
        
        # Send invite email
        send_invite_email(
            to_email=request.email,
            full_name=request.full_name,
            invite_token=raw_token,
            organisation_name=current_admin.tenant.name
        )
        
        return {"message": "Invite sent successfully"}
    except HTTPException as he:
        # Re-raise HTTPException raised by email service
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/accept-invite", response_model=MessageResponse)
def accept_invite(request: AcceptInviteRequest, db: Session = Depends(get_db)):
    """Open route to activate an invited member account using a valid invite token."""
    hashed_token = hash_token(request.token)
    invite_token = db.query(InviteToken).filter(InviteToken.token_hash == hashed_token).first()
    
    if not invite_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid invite token"
        )
    if invite_token.is_used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invite token has already been used"
        )
        
    current_time = datetime.now(timezone.utc)
    expires_at = invite_token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
        
    if expires_at < current_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invite token has expired"
        )
        
    user = invite_token.user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User associated with token not found"
        )
        
    try:
        user.hashed_password = hash_password(request.password)
        user.is_active = True
        invite_token.is_used = True
        db.commit()
        return {"message": "Account activated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Protected route to get the current authenticated user's details."""
    return current_user

@router.get("/google")
def google_auth_init():
    """Redirects the client to Google's OAuth consent screen."""
    return RedirectResponse(url=get_google_auth_url(), status_code=302)

@router.get("/google/callback")
async def google_auth_callback(
    code: str | None = None,
    state: str | None = None,
    db: Session = Depends(get_db)
):
    """Callback route for Google OAuth redirect."""
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code missing"
        )
    
    user_info = await exchange_code_for_user_info(code)
    google_id = user_info["google_id"]
    email = user_info["email"]
    full_name = user_info["full_name"]
    avatar_url = user_info["avatar_url"]
    
    user = db.query(User).options(joinedload(User.role)).filter(
        (User.google_id == google_id) | (User.email == email)
    ).first()
    
    if user:
        if not user.google_id:
            user.google_id = google_id
            if avatar_url:
                user.avatar_url = avatar_url
            db.commit()
            db.refresh(user)
            user = db.query(User).options(joinedload(User.role)).filter(User.id == user.id).first()
            
        if not user.is_active:
            return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error=inactive")
            
        # Generate tokens
        access_token = create_access_token({
            "sub": str(user.id),
            "tenant_id": str(user.tenant_id),
            "role_id": str(user.role_id),
            "is_admin": user.role.is_admin
        })
        raw_refresh_token, hashed_refresh_token = create_refresh_token()
        
        expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        db_refresh_token = RefreshToken(
            user_id=user.id,
            token_hash=hashed_refresh_token,
            expires_at=expires_at,
            is_revoked=False
        )
        db.add(db_refresh_token)
        db.commit()
        
        response = RedirectResponse(url=f"{settings.FRONTEND_URL}/dashboard")
        
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            httponly=True,
            samesite="lax",
            secure=False
        )
        response.set_cookie(
            key="refresh_token",
            value=raw_refresh_token,
            max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
            secure=False
        )
        return response
    else:
        token = create_google_setup_token(email, full_name, google_id, avatar_url)
        return RedirectResponse(url=f"{settings.FRONTEND_URL}/register/google-org?setup_token={token}")

@router.post("/google/complete-setup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def google_complete_setup(
    request: GoogleOrgSetupRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Verifies setup token and performs single-transaction creation of Tenant and User."""
    decoded = verify_google_setup_token(request.setup_token)
    email = decoded["email"]
    full_name = decoded["full_name"]
    google_id = decoded["google_id"]
    avatar_url = decoded.get("avatar_url")
    
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists"
        )
        
    try:
        org_name = request.org_name
        base_slug = re.sub(r'[^a-z0-9\-]', '', org_name.lower().replace(" ", "-"))
        slug = base_slug or "tenant"
        while db.query(Tenant).filter(Tenant.slug == slug).first():
            slug = f"{base_slug}-{random.randint(1000, 9999)}"

        tenant = Tenant(
            name=org_name,
            slug=slug
        )
        db.add(tenant)
        db.flush()

        # Create Admin default role for this tenant
        admin_role = Role(
            tenant_id=tenant.id,
            name="Admin",
            is_admin=True,
            is_default=True
        )
        db.add(admin_role)

        # Create Member default role for this tenant
        member_role = Role(
            tenant_id=tenant.id,
            name="Member",
            is_admin=False,
            is_default=True
        )
        db.add(member_role)
        db.flush()

        user = User(
            tenant_id=tenant.id,
            email=email,
            full_name=full_name,
            hashed_password="",
            role_id=admin_role.id,
            is_active=True,
            google_id=google_id,
            avatar_url=avatar_url,
            has_password=False
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        # Eager load user.role
        user = db.query(User).options(joinedload(User.role)).filter(User.id == user.id).first()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # Generate tokens
    access_token = create_access_token({
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "role_id": str(user.role_id),
        "is_admin": user.role.is_admin
    })
    raw_refresh_token, hashed_refresh_token = create_refresh_token()
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db_refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=hashed_refresh_token,
        expires_at=expires_at,
        is_revoked=False
    )
    db.add(db_refresh_token)
    db.commit()

    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )
    response.set_cookie(
        key="refresh_token",
        value=raw_refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=False
    )
    return user

@router.post("/set-password", response_model=MessageResponse)
def set_password(
    request: SetPasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Protected endpoint to set user password if they don't have one."""
    if current_user.has_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have a password set. Use forgot password to change it."
        )
    
    try:
        current_user.hashed_password = hash_password(request.new_password)
        current_user.has_password = True
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    return {"message": "Password set successfully."}
