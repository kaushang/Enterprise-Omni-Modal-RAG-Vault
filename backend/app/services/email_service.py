import resend
from app.core.config import settings
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def send_invite_email(to_email: str, full_name: str, invite_token: str, organisation_name: str) -> None:
    """Sends an invitation email using the Resend Python SDK.
    If sending fails, logs the error and raises an HTTPException.
    """
    resend.api_key = settings.RESEND_API_KEY
    print("🩷🩷")
    print("resend.api_key: ", resend.api_key)
    print("🩷🩷")
    print("settings.RESEND_API_KEY: ", settings.RESEND_API_KEY)
    invite_url = f"{settings.FRONTEND_URL}/accept-invite?token={invite_token}"
    
    if settings.RESEND_API_KEY == "API_KEY":
        logger.warning(f"[DEVELOPMENT] Resend API key is not configured. Email not sent. Invite URL: {invite_url}")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Invitation to join RAG Vault</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f3f4f6;
                padding: 20px;
                margin: 0;
            }}
            .card {{
                max-width: 600px;
                margin: 0 auto;
                background: #ffffff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            }}
            .button {{
                display: inline-block;
                background-color: #4f46e5;
                color: #ffffff !important;
                padding: 12px 24px;
                text-decoration: none;
                font-weight: bold;
                border-radius: 6px;
                margin: 20px 0;
            }}
            .footer {{
                font-size: 0.875rem;
                color: #6b7280;
                margin-top: 20px;
                border-top: 1px solid #e5e7eb;
                padding-top: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>You've been invited!</h2>
            <p>Hello {full_name},</p>
            <p>You have been invited to join the organisation <strong>{organisation_name}</strong> on Enterprise OmniModal RAG Vault.</p>
            <p>Please click the button below to accept your invitation and activate your account. This invitation will expire in 48 hours.</p>
            
            <a href="{invite_url}" class="button">Accept Invitation</a>
            
            <p>If the button doesn't work, copy and paste this URL into your browser:</p>
            <p style="word-break: break-all;"><a href="{invite_url}">{invite_url}</a></p>
            
            <div class="footer">
                This is an automated message. Please do not reply directly to this email.
            </div>
        </div>
    </body>
    </html>
    """
    
    try:
        params = {
            "from": "noreply@grooviamusic.com", # This email is temporary and only for development purpose. This must be changed before production
            "to": to_email,
            "subject": f"You have been invited to join {organisation_name} on RAG Vault",
            "html": html_content
        }
        resend.Emails.send(params)
    except Exception as e:
        logger.error(f"Failed to send invite email to {to_email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send invitation email: {str(e)}"
        )

def send_otp_email(to_email: str, full_name: str, otp: str, org_name: str) -> None:
    """Sends a registration OTP verification email using the Resend Python SDK.
    If sending fails, logs the error and raises an HTTPException.
    """
    resend.api_key = settings.RESEND_API_KEY
    first_name = full_name.split()[0] if full_name else "User"

    if settings.RESEND_API_KEY == "API_KEY" or not settings.RESEND_API_KEY:
        logger.warning(f"[DEVELOPMENT] Resend API key is not configured. Registration OTP email not sent. to={to_email}, otp={otp}")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Your verification code for RAG Vault</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f3f4f6;
                padding: 20px;
                margin: 0;
            }}
            .card {{
                max-width: 600px;
                margin: 0 auto;
                background: #ffffff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            }}
            .otp-box {{
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                letter-spacing: 8px;
                margin: 30px 0;
                padding: 15px;
                background-color: #f9fafb;
                border: 1px dashed #d1d5db;
                border-radius: 6px;
                color: #111827;
            }}
            .footer {{
                font-size: 0.875rem;
                color: #6b7280;
                margin-top: 20px;
                border-top: 1px solid #e5e7eb;
                padding-top: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Your Verification Code</h2>
            <p>Hello {first_name},</p>
            <p>Thank you for registering. Please use the following 6-digit verification code to complete your registration for <strong>{org_name}</strong>:</p>
            
            <div class="otp-box">{otp}</div>
            
            <p>This code is valid for <strong>10 minutes</strong>. If you did not request this, you can safely ignore this email.</p>
            
            <div class="footer">
                This is an automated message. Please do not reply directly to this email.
            </div>
        </div>
    </body>
    </html>
    """

    try:
        params = {
            "from": "onboarding@resend.dev",
            "to": to_email,
            "subject": "Your verification code for RAG Vault",
            "html": html_content
        }
        resend.Emails.send(params)
    except Exception as e:
        logger.error(f"Failed to send OTP email to {to_email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send OTP email: {str(e)}"
        )

def send_forgot_password_otp_email(to_email: str, full_name: str, otp: str) -> None:
    """Sends a forgot password OTP email using the Resend Python SDK.
    If sending fails, logs the error and raises an HTTPException.
    """
    resend.api_key = settings.RESEND_API_KEY
    first_name = full_name.split()[0] if full_name else "User"

    if settings.RESEND_API_KEY == "API_KEY" or not settings.RESEND_API_KEY:
        logger.warning(f"[DEVELOPMENT] Resend API key is not configured. Forgot password OTP email not sent. to={to_email}, otp={otp}")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Reset your RAG Vault password</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f3f4f6;
                padding: 20px;
                margin: 0;
            }}
            .card {{
                max-width: 600px;
                margin: 0 auto;
                background: #ffffff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            }}
            .otp-box {{
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                letter-spacing: 8px;
                margin: 30px 0;
                padding: 15px;
                background-color: #f9fafb;
                border: 1px dashed #d1d5db;
                border-radius: 6px;
                color: #111827;
            }}
            .footer {{
                font-size: 0.875rem;
                color: #6b7280;
                margin-top: 20px;
                border-top: 1px solid #e5e7eb;
                padding-top: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Reset Your Password</h2>
            <p>Hello {first_name},</p>
            <p>A password reset was requested for your RAG Vault account. Please use the following 6-digit verification code to reset your password:</p>
            
            <div class="otp-box">{otp}</div>
            
            <p>This code is valid for <strong>10 minutes</strong>. If you did not request this, you should ignore this email and your password will not be changed.</p>
            
            <div class="footer">
                This is an automated message. Please do not reply directly to this email.
            </div>
        </div>
    </body>
    </html>
    """

    try:
        params = {
            "from": "onboarding@resend.dev",
            "to": to_email,
            "subject": "Reset your RAG Vault password",
            "html": html_content
        }
        resend.Emails.send(params)
    except Exception as e:
        logger.error(f"Failed to send forgot password email to {to_email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send forgot password email: {str(e)}"
        )
