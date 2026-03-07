/**
 * components/volunteer/VolunteerHeader.tsx
 *
 * Hero header for the volunteer page.
 * Shows sign-in/sign-up CTAs for guests, and a user badge for logged-in users.
 */

import type { Profile } from '@/lib/supabase';

type Props = {
  currentUser: any;
  profile: Profile | null;
  authLoading: boolean;
  onSignUpClick: () => void;
  onSignInClick: () => void;
  onSignOut: () => void;
};

export default function VolunteerHeader({
  currentUser, profile, authLoading,
  onSignUpClick, onSignInClick, onSignOut,
}: Props) {
  return (
    <section className="relative bg-linear-to-br from-green-800 to-emerald-900 text-white py-24 px-4 overflow-hidden">
      {/* Dot grid background */}
      <div
        className="absolute inset-0 opacity-10"
        style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)', backgroundSize: '32px 32px' }}
      />
      <div className="absolute top-0 right-0 w-80 h-80 bg-emerald-400 opacity-10 rounded-full translate-x-1/3 -translate-y-1/3 blur-3xl" />

      <div className="max-w-4xl mx-auto text-center relative z-10">
        <span className="inline-block bg-emerald-600 text-emerald-100 text-sm font-medium px-4 py-1 rounded-full mb-6">
          🤝 Join the Movement
        </span>
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 leading-tight">
          Volunteer &<br />
          <span className="text-emerald-300">Organizer Hub</span>
        </h1>
        <p className="text-xl text-emerald-100 max-w-2xl mx-auto leading-relaxed mb-8">
          Be part of Davao City's clean-up revolution. Join drives as a volunteer,
          upload event photos, or step up as an organizer to lead your community.
        </p>

        {/* Guest CTAs */}
        {!authLoading && !currentUser && (
          <div className="flex gap-4 justify-center">
            <button
              onClick={onSignUpClick}
              className="bg-emerald-400 text-green-950 font-bold px-8 py-3 rounded-full hover:bg-emerald-300 transition-colors"
            >
              Sign Up to Volunteer
            </button>
            <button
              onClick={onSignInClick}
              className="border-2 border-emerald-400 text-emerald-200 px-8 py-3 rounded-full hover:bg-emerald-800 transition-colors"
            >
              Sign In
            </button>
          </div>
        )}

        {/* Logged-in user badge */}
        {!authLoading && currentUser && (
          <div className="inline-flex items-center gap-3 bg-emerald-700 bg-opacity-60 rounded-2xl px-6 py-3">
            <div className="w-9 h-9 bg-emerald-400 rounded-full flex items-center justify-center text-green-950 font-bold text-sm">
              {currentUser.email?.[0]?.toUpperCase()}
            </div>
            <div className="text-left">
              <div className="text-sm font-semibold">{currentUser.email}</div>
              <div className="text-xs text-emerald-300 capitalize">{profile?.role ?? 'volunteer'}</div>
            </div>
            <button
              onClick={onSignOut}
              className="ml-4 text-xs text-emerald-300 hover:text-white border border-emerald-500 px-3 py-1 rounded-full"
            >
              Sign Out
            </button>
          </div>
        )}
      </div>
    </section>
  );
}