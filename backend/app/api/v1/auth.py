from datetime import datetime, timedelta, timezone
import secrets
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.config import settings
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_token
)
from app.core.dependencies import get_current_user, require_admin
from app.models.user import User
from app.models.tenant import Tenant
from app.models.refresh_token import RefreshToken
from app.models.invite_token import InviteToken
from app.models.enums import UserRole
from app.schemas.auth import (
    AdminRegisterRequest,
    LoginRequest,
    AcceptInviteRequest,
    UserResponse,
    InviteMemberRequest,
    MessageResponse
)
from app.services.email_service import send_invite_email

router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(request: AdminRegisterRequest, db: Session = Depends(get_db)):
    """Open route to register an admin user and create a new tenant in a single transaction."""
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
        
    try:
        # Generate slug by lowercasing and replacing spaces with hyphens
        slug = request.organisation_name.lower().replace(" ", "-")
        
        tenant = Tenant(
            name=request.organisation_name,
            slug=slug
        )
        db.add(tenant)
        db.flush()  # Obtain tenant.id
        
        user = User(
            tenant_id=tenant.id,
            email=request.email,
            full_name=request.full_name,
            hashed_password=hash_password(request.password),
            role=UserRole.admin,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=UserResponse)
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """Open route to log in, setting HTTP-only cookies on success."""
    user = db.query(User).filter(User.email == request.email).first()
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
        "role": user.role.value
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
        
    user = db_token.user
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
        
    # Generate new access token
    access_token = create_access_token({
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "role": user.role.value
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
            role=request.role,
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
