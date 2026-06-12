export interface RoleResponse {
  id: string
  name: string
  is_admin: boolean
  is_default: boolean
  tenant_id: string
  created_at: string
}

export interface UserResponse {
  id: string
  email: string
  full_name: string
  role_id: string
  role: RoleResponse
  tenant_id: string
  is_active: boolean
  has_password: boolean
  avatar_url: string | null
  created_at: string
}

export interface MessageResponse {
  message: string
}

export interface RegisterSignupPayload {
  org_name: string
  org_website: string
  full_name: string
  email: string
  password: string
  confirm_password: string
}

export interface VerifyOTPPayload {
  email: string
  otp: string
}

export interface LoginPayload {
  email: string
  password: string
}

export interface ForgotPasswordPayload {
  email: string
}

export interface ResetPasswordPayload {
  email: string
  otp: string
  new_password: string
  confirm_password: string
}

export interface AcceptInvitePayload {
  token: string
  password: string
}

export interface GoogleOrgSetupPayload {
  org_name: string
  org_website: string
  setup_token: string
}

export interface SetPasswordPayload {
  new_password: string
  confirm_password: string
}
