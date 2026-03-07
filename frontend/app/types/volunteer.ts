/**
 * types/volunteer.ts
 * Shared types for the volunteer/organizer feature.
 */

import type { Event, Participant, Upload } from '@/lib/supabase';

/** Event enriched with organizer email, participant count, and current user's participation */
export type EventWithMeta = Event & {
  organizer_email?: string;
  participant_count?: number;
  user_participation?: Participant | null;
};

/** Controls whether the event form modal is open and in which mode */
export type EventModalMode = 'create' | 'edit' | null;

/** Fields for the create/edit event form */
export type EventFormState = {
  title: string;
  description: string;
  location: string;
  event_date: string;
};

export const EMPTY_FORM: EventFormState = {
  title: '',
  description: '',
  location: '',
  event_date: '',
};

/** Participant row enriched with the volunteer's email from profiles */
export type ParticipantWithEmail = Participant & { email?: string };

/** Upload row enriched with the uploader's email from profiles */
export type UploadWithEmail = Upload & { email?: string };