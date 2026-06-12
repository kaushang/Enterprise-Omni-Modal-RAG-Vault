import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { logout as apiLogout } from '../services/authService';
import { Shield, LogOut, User, Building, Mail, Award, CheckCircle } from 'lucide-react';

const DashboardPage: React.FC = () => {
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await apiLogout();
    } catch (err) {
      console.error('Logout API call failed, removing local session anyway', err);
    } finally {
      logout();
      setIsLoggingOut(false);
      navigate('/login');
    }
  };

  if (!user) return null;

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* Top Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center gap-3">
              <Shield className="h-8 w-8 text-indigo-700" />
              <span className="font-sora text-xl font-bold text-slate-800">
                RAG Vault
              </span>
            </div>
            <div className="flex items-center">
              <button
                onClick={handleLogout}
                disabled={isLoggingOut}
                className="inline-flex items-center gap-2 border border-slate-200 text-slate-700 bg-white hover:bg-slate-50 font-semibold rounded-lg px-4 py-2 text-sm transition-colors duration-200"
              >
                <LogOut className="h-4 w-4" />
                {isLoggingOut ? 'Signing out...' : 'Sign out'}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        {/* Welcome Card */}
        <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm mb-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="h-16 w-16 rounded-full bg-indigo-50 flex items-center justify-center text-indigo-700">
              <User className="h-8 w-8" />
            </div>
            <div>
              <h1 className="font-sora text-2xl font-bold text-slate-800">
                Welcome back, {user.full_name}!
              </h1>
              <p className="text-slate-500 text-sm">
                Your enterprise session is active and secured.
              </p>
            </div>
          </div>

          <div className="border-t border-slate-100 pt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* User Email */}
              <div className="flex items-center gap-3">
                <Mail className="h-5 w-5 text-slate-400 shrink-0" />
                <div>
                  <span className="block text-xs font-semibold text-slate-400 uppercase tracking-wider">Email Address</span>
                  <span className="text-slate-800 text-sm font-medium">{user.email}</span>
                </div>
              </div>

              {/* Organization */}
              <div className="flex items-center gap-3">
                <Building className="h-5 w-5 text-slate-400 shrink-0" />
                <div>
                  <span className="block text-xs font-semibold text-slate-400 uppercase tracking-wider">Tenant ID</span>
                  <span className="text-slate-800 text-sm font-medium font-mono">{user.tenant_id}</span>
                </div>
              </div>

              {/* Role */}
              <div className="flex items-center gap-3">
                <Award className="h-5 w-5 text-slate-400 shrink-0" />
                <div>
                  <span className="block text-xs font-semibold text-slate-400 uppercase tracking-wider">Role / Access Level</span>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 mt-0.5 capitalize">
                    {user.role.name}
                  </span>
                </div>
              </div>

              {/* Status */}
              <div className="flex items-center gap-3">
                <CheckCircle className="h-5 w-5 text-slate-400 shrink-0" />
                <div>
                  <span className="block text-xs font-semibold text-slate-400 uppercase tracking-wider">Account Status</span>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 mt-0.5">
                    Active
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Info Banner */}
        <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-6 text-indigo-900 text-sm leading-relaxed">
          <h3 className="font-semibold text-indigo-950 mb-1">Interactive Sandbox Environment</h3>
          <p>
            You are currently authenticated in the Enterprise Omni-Modal RAG Vault frontend. You can use the navbar to log out or refresh the page to verify that the session persists correctly.
          </p>
        </div>
      </main>
    </div>
  );
};

export default DashboardPage;
