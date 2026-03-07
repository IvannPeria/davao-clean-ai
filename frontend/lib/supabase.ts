/**
 * lib/supabase.ts
 *
 * Supabase client singleton used across the entire app.
 *
 * SETUP INSTRUCTIONS:
 * 1. Your .env.local file (under the project root, NOT inside /app) must contain:
 *      NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
 *      NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
 *
 *    ⚠️  If your .env file is currently inside /app, move it to the project ROOT
 *        (same level as package.json). Next.js only reads .env from the root.
 *
 * 2. Install the Supabase JS client if you haven't already:
 *      npm install @supabase/supabase-js
 *
 * 3. For Auth (sign in / sign up), you'll also want:
 *      npm install @supabase/auth-helpers-nextjs
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    'Missing Supabase environment variables. ' +
    'Ensure NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY are set in .env.local at the project root.'
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// ─── TypeScript types matching your Supabase schema ──────────────────────────

export type Profile = {
  id: string;           // uuid — links to auth.users.id
  email: string;
  role: string;         // 'volunteer' | 'organizer'
  updated_at: string;
};

export type Event = {
  id: string;           // uuid
  organizer_id: string; // uuid → profiles.id
  title: string;
  description: string;
  location: string;
  event_date: string;   // timestamptz
  created_at: string;
};

export type Participant = {
  id: string;           // uuid
  event_id: string;     // uuid → events.id
  volunteer_id: string; // uuid → profiles.id
  status: string;       // 'pending' | 'accepted' | 'declined'
};

export type Upload = {
  id: string;           // uuid
  user_id: string;      // uuid → profiles.id
  image_url: string;
  category: string;     // waste category from classifier, OR event_id reference
  created_at: string;
};