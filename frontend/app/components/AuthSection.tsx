/**
 * components/volunteer/AuthSection.tsx
 *
 * Sign in / sign up form shown to guests at the bottom of the volunteer page.
 * Toggling between modes is handled locally; actual auth is delegated via props.
 */

type Props = {
  authMode: 'signin' | 'signup';
  authEmail: string;
  authPassword: string;
  authError: string;
  authFormLoading: boolean;
  onEmailChange: (val: string) => void;
  onPasswordChange: (val: string) => void;
  onSignIn: () => void;
  onSignUp: () => void;
  onToggleMode: () => void;
};

export default function AuthSection({
  authMode, authEmail, authPassword, authError, authFormLoading,
  onEmailChange, onPasswordChange, onSignIn, onSignUp, onToggleMode,
}: Props) {
  return (
    <section id="auth-section" className="py-24 px-4 bg-white">
      <div className="max-w-md mx-auto">
        <div className="bg-white border border-green-200 rounded-2xl p-8 shadow-sm">
          <h2 className="text-2xl font-bold text-gray-900 mb-2 text-center">
            {authMode === 'signin' ? 'Sign In' : 'Create Account'}
          </h2>
          <p className="text-gray-500 text-sm text-center mb-6">
            {authMode === 'signin'
              ? 'Sign in to join events and upload photos.'
              : 'Create a free account to start volunteering.'}
          </p>

          {authError && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-xl mb-4">
              {authError}
            </div>
          )}

          <div className="space-y-4">
            <input
              type="email"
              placeholder="Email address"
              value={authEmail}
              onChange={(e) => onEmailChange(e.target.value)}
              className="w-full border bg-white border-gray-200 rounded-xl px-4 py-3 text-gray-500 text-sm placeholder:text-gray-400 focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
            />
            <input
              type="password"
              placeholder="Password"
              value={authPassword}
              onChange={(e) => onPasswordChange(e.target.value)}
              className="w-full border bg-white border-gray-200 rounded-xl px-4 py-3 text-gray-500 text-sm placeholder:text-gray-400 focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
            />
            <button
              onClick={authMode === 'signin' ? onSignIn : onSignUp}
              disabled={authFormLoading}
              className="w-full bg-green-600 text-white py-3 rounded-xl font-semibold hover:bg-green-700 disabled:opacity-60 transition-colors"
            >
              {authFormLoading
                ? 'Please wait...'
                : authMode === 'signin' ? 'Sign In' : 'Create Account'}
            </button>
          </div>

          <p className="text-center text-sm text-gray-500 mt-4">
            {authMode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
            <button
              onClick={onToggleMode}
              className="text-green-600 font-medium hover:underline"
            >
              {authMode === 'signin' ? 'Sign up free' : 'Sign in'}
            </button>
          </p>
        </div>
      </div>
    </section>
  );
}