'use client';

/**
 * hooks/useVolunteer.ts
 * All Supabase logic and state for the volunteer/organizer page.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { supabase } from '@/lib/supabase';
import type { Profile, Event, Participant, Upload } from '@/lib/supabase';
import {
  EMPTY_FORM,
  type EventWithMeta,
  type EventModalMode,
  type EventFormState,
  type ParticipantWithEmail,
  type UploadWithEmail,
} from '../types/volunteer';

export function useVolunteer() {
  // ── Auth & Profile ──────────────────────────────────────────────────────────
  const [currentUser, setCurrentUser]   = useState<any>(null);
  const [profile, setProfile]           = useState<Profile | null>(null);
  const [authLoading, setAuthLoading]   = useState(true);

  // Stored in state (not derived inline) so UI re-renders immediately on role change
  const [isOrganizer, setIsOrganizer] = useState(false);

  // ── Auth Form ───────────────────────────────────────────────────────────────
  const [authMode, setAuthMode]             = useState<'signin' | 'signup'>('signin');
  const [authEmail, setAuthEmail]           = useState('');
  const [authPassword, setAuthPassword]     = useState('');
  const [authError, setAuthError]           = useState('');
  const [authFormLoading, setAuthFormLoading] = useState(false);

  // ── Events ──────────────────────────────────────────────────────────────────
  const [events, setEvents]               = useState<EventWithMeta[]>([]);
  const [eventsLoading, setEventsLoading] = useState(true);

  // ── Event Detail Modal ──────────────────────────────────────────────────────
  const [selectedEvent, setSelectedEvent]       = useState<EventWithMeta | null>(null);
  const [eventParticipants, setEventParticipants] = useState<ParticipantWithEmail[]>([]);
  const [eventUploads, setEventUploads]         = useState<UploadWithEmail[]>([]);
  const [detailLoading, setDetailLoading]       = useState(false);

  // ── Event CRUD Form Modal ───────────────────────────────────────────────────
  const [eventModalMode, setEventModalMode] = useState<EventModalMode>(null);
  const [editingEvent, setEditingEvent]     = useState<Event | null>(null);
  const [eventForm, setEventForm]           = useState<EventFormState>(EMPTY_FORM);
  const [formLoading, setFormLoading]       = useState(false);
  const [formError, setFormError]           = useState('');

  // ── Image Upload ────────────────────────────────────────────────────────────
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError]     = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Role ────────────────────────────────────────────────────────────────────
  const [roleLoading, setRoleLoading] = useState(false);

  // ─── Fetch & sync profile ────────────────────────────────────────────────────
  const fetchProfile = useCallback(async (userId: string) => {
    const { data } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();
    if (data) {
      setProfile(data as Profile);
      setIsOrganizer((data as Profile).role === 'organizer');
    }
  }, []);

  // ─── Auth listener ────────────────────────────────────────────────────────
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setCurrentUser(session?.user ?? null);
      if (session?.user) fetchProfile(session.user.id);
      setAuthLoading(false);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setCurrentUser(session?.user ?? null);
      if (session?.user) fetchProfile(session.user.id);
      else { setProfile(null); setIsOrganizer(false); }
    });

    return () => listener.subscription.unsubscribe();
  }, [fetchProfile]);

  // ─── Fetch all events ─────────────────────────────────────────────────────
  const fetchEvents = useCallback(async () => {
    setEventsLoading(true);

    const { data: eventsData, error } = await supabase
      .from('events')
      .select('*')
      .order('event_date', { ascending: true });

    if (error || !eventsData) { setEventsLoading(false); return; }

    const enriched: EventWithMeta[] = await Promise.all(
      eventsData.map(async (ev: Event) => {
        const { data: org } = await supabase
          .from('profiles').select('email').eq('id', ev.organizer_id).single();

        // Live count — always re-fetched
        const { count } = await supabase
          .from('participants')
          .select('*', { count: 'exact', head: true })
          .eq('event_id', ev.id);

        // Current user's participation row for this event
        let userPart: Participant | null = null;
        if (currentUser) {
          const { data: p } = await supabase
            .from('participants').select('*')
            .eq('event_id', ev.id)
            .eq('volunteer_id', currentUser.id)
            .maybeSingle();
          userPart = p ?? null;
        }

        return {
          ...ev,
          organizer_email: org?.email ?? 'Unknown',
          participant_count: count ?? 0,
          user_participation: userPart,
        };
      })
    );

    setEvents(enriched);
    setEventsLoading(false);
  }, [currentUser]);

  useEffect(() => { fetchEvents(); }, [fetchEvents]);

  // ─── Fetch event detail ────────────────────────────────────────────────────
  const fetchEventDetail = useCallback(async (event: EventWithMeta) => {
    setDetailLoading(true);
    setSelectedEvent(event);

    const { data: parts } = await supabase
      .from('participants').select('*').eq('event_id', event.id)
      .order('status', { ascending: true });

    const enrichedParts: ParticipantWithEmail[] = await Promise.all(
      (parts ?? []).map(async (p: Participant) => {
        const { data: prof } = await supabase
          .from('profiles').select('email').eq('id', p.volunteer_id).single();
        return { ...p, email: prof?.email ?? 'Unknown' };
      })
    );
    setEventParticipants(enrichedParts);

    const { data: ups } = await supabase
      .from('uploads').select('*').eq('event_id', event.id)
      .order('created_at', { ascending: false });

    const enrichedUps: UploadWithEmail[] = await Promise.all(
      (ups ?? []).map(async (u: Upload) => {
        const { data: prof } = await supabase
          .from('profiles').select('email').eq('id', u.user_id).single();
        return { ...u, email: prof?.email ?? 'Unknown' };
      })
    );
    setEventUploads(enrichedUps);
    setDetailLoading(false);
  }, []);

  // ─── Auth handlers ────────────────────────────────────────────────────────
  const handleSignUp = async () => {
    setAuthFormLoading(true);
    setAuthError('');
    const { data, error } = await supabase.auth.signUp({ email: authEmail, password: authPassword });
    if (error) { setAuthError(error.message); setAuthFormLoading(false); return; }
    if (data.user) {
      await supabase.from('profiles').insert({ id: data.user.id, email: authEmail, role: 'volunteer' });
    }
    setAuthFormLoading(false);
  };

  const handleSignIn = async () => {
    setAuthFormLoading(true);
    setAuthError('');
    const { error } = await supabase.auth.signInWithPassword({ email: authEmail, password: authPassword });
    if (error) setAuthError(error.message);
    setAuthFormLoading(false);
  };

  const handleSignOut = async () => { await supabase.auth.signOut(); };

  // ─── Role switching — both directions ─────────────────────────────────────
  /**
   * Shared internal function that updates role in Supabase.
   * Tries direct UPDATE first; falls back to RPC (SECURITY DEFINER) if blocked.
   */
  const switchRole = async (newRole: 'organizer' | 'volunteer') => {
    if (!currentUser) return;
    setRoleLoading(true);

    const { error: updateError } = await supabase
      .from('profiles')
      .update({ role: newRole })
      .eq('id', currentUser.id);

    if (updateError) {
      // Fallback RPC — run upgrade_to_organizer or downgrade_to_volunteer in Supabase
      const rpcName = newRole === 'organizer' ? 'upgrade_to_organizer' : 'downgrade_to_volunteer';
      const { error: rpcError } = await supabase.rpc(rpcName);
      if (rpcError) {
        console.error(`Role switch to ${newRole} failed:`, rpcError.message);
        setRoleLoading(false);
        return;
      }
    }

    await fetchProfile(currentUser.id);
    setRoleLoading(false);
  };

  const handleBecomeOrganizer = () => switchRole('organizer');
  const handleBecomeVolunteer = () => switchRole('volunteer');

  // ─── Participant CRUD ──────────────────────────────────────────────────────
  /**
   * Join an event. Works for both volunteers AND organizers joining other events.
   * Logs the actual error to console so silent failures are visible during dev.
   */
  const handleJoinEvent = async (eventId: string) => {
    if (!currentUser) return;

    // Guard: prevent joining own event
    const ownEvent = events.find(e => e.id === eventId && e.organizer_id === currentUser.id);
    if (ownEvent) return;

    // Guard: prevent duplicate join
    const { data: existing } = await supabase
      .from('participants').select('id')
      .eq('event_id', eventId)
      .eq('volunteer_id', currentUser.id)
      .maybeSingle();
    if (existing) return;

    const { error } = await supabase.from('participants').insert({
      event_id: eventId,
      volunteer_id: currentUser.id,
      status: 'accepted',
    });

    if (error) {
      // Surface the real Supabase error — most likely an RLS policy block
      console.error('Join failed:', error.message, error.details, error.hint);
      return;
    }

    await fetchEvents();
    if (selectedEvent?.id === eventId) await fetchEventDetail(selectedEvent);
  };

  const handleLeaveEvent = async (eventId: string) => {
    if (!currentUser) return;
    const { error } = await supabase
      .from('participants').delete()
      .eq('event_id', eventId)
      .eq('volunteer_id', currentUser.id);

    if (error) { console.error('Leave failed:', error.message); return; }

    await fetchEvents();
    if (selectedEvent?.id === eventId) await fetchEventDetail(selectedEvent);
  };

  const handleUpdateParticipantStatus = async (participantId: string, status: 'accepted' | 'declined') => {
    await supabase.from('participants').update({ status }).eq('id', participantId);
    if (selectedEvent) await fetchEventDetail(selectedEvent);
    await fetchEvents();
  };

  const handleRemoveParticipant = async (participantId: string) => {
    await supabase.from('participants').delete().eq('id', participantId);
    if (selectedEvent) await fetchEventDetail(selectedEvent);
    await fetchEvents();
  };

  // ─── Event CRUD ───────────────────────────────────────────────────────────
  const openCreateModal = () => {
    setEventForm(EMPTY_FORM);
    setFormError('');
    setEditingEvent(null);
    setEventModalMode('create');
  };

  const openEditModal = (ev: Event) => {
    setEventForm({
      title: ev.title,
      description: ev.description,
      location: ev.location,
      event_date: ev.event_date.slice(0, 16),
    });
    setFormError('');
    setEditingEvent(ev);
    setEventModalMode('edit');
  };

  const handleEventFormSubmit = async () => {
    if (!currentUser) return;
    setFormLoading(true);
    setFormError('');

    if (!eventForm.title || !eventForm.location || !eventForm.event_date) {
      setFormError('Title, location, and date are required.');
      setFormLoading(false);
      return;
    }

    if (eventModalMode === 'create') {
      const { error } = await supabase.from('events').insert({
        organizer_id: currentUser.id,
        title: eventForm.title,
        description: eventForm.description,
        location: eventForm.location,
        event_date: new Date(eventForm.event_date).toISOString(),
      });
      if (error) { setFormError(error.message); setFormLoading(false); return; }

    } else if (eventModalMode === 'edit' && editingEvent) {
      const { error } = await supabase.from('events').update({
        title: eventForm.title,
        description: eventForm.description,
        location: eventForm.location,
        event_date: new Date(eventForm.event_date).toISOString(),
      }).eq('id', editingEvent.id);
      if (error) { setFormError(error.message); setFormLoading(false); return; }
    }

    setEventModalMode(null);
    setFormLoading(false);
    await fetchEvents();
  };

  const handleDeleteEvent = async (eventId: string) => {
    if (!confirm('Delete this event? This cannot be undone.')) return;
    await supabase.from('events').delete().eq('id', eventId);
    if (selectedEvent?.id === eventId) setSelectedEvent(null);
    await fetchEvents();
  };

  // ─── Image upload ──────────────────────────────────────────────────────────
  const handleImageUpload = async (file: File) => {
    if (!currentUser || !selectedEvent) return;
    setUploadLoading(true);
    setUploadError('');

    const filePath = `${selectedEvent.id}/${currentUser.id}/${Date.now()}_${file.name}`;

    const { error: storageError } = await supabase.storage
      .from('event-images').upload(filePath, file, { upsert: false });

    if (storageError) { setUploadError(storageError.message); setUploadLoading(false); return; }

    const { data: urlData } = supabase.storage.from('event-images').getPublicUrl(filePath);

    const { error: dbError } = await supabase.from('uploads').insert({
      user_id: currentUser.id,
      image_url: urlData.publicUrl,
      category: 'event',
      event_id: selectedEvent.id,
    });

    if (dbError) { setUploadError(dbError.message); setUploadLoading(false); return; }

    setUploadLoading(false);
    await fetchEventDetail(selectedEvent);
  };

  const handleDeleteUpload = async (upload: Upload) => {
    if (!currentUser || upload.user_id !== currentUser.id) return;
    await supabase.from('uploads').delete().eq('id', upload.id);
    if (selectedEvent) await fetchEventDetail(selectedEvent);
  };

  return {
    // Auth
    currentUser, profile, authLoading,
    authMode, setAuthMode,
    authEmail, setAuthEmail,
    authPassword, setAuthPassword,
    authError, authFormLoading,
    handleSignIn, handleSignUp, handleSignOut,
    // Role
    isOrganizer, roleLoading,
    handleBecomeOrganizer, handleBecomeVolunteer,
    // Events
    events, eventsLoading,
    isMyEvent: (ev: Event) => !!(currentUser && ev.organizer_id === currentUser.id),
    fetchEventDetail,
    handleJoinEvent, handleLeaveEvent,
    handleDeleteEvent,
    // Event form modal
    eventModalMode, setEventModalMode,
    eventForm, setEventForm,
    formError, formLoading,
    openCreateModal, openEditModal,
    handleEventFormSubmit,
    // Event detail modal
    selectedEvent, setSelectedEvent,
    eventParticipants, eventUploads,
    detailLoading,
    handleUpdateParticipantStatus,
    handleRemoveParticipant,
    // Uploads
    uploadLoading, uploadError, setUploadError,
    fileInputRef,
    handleImageUpload, handleDeleteUpload,
  };
}