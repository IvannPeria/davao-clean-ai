/**
 * components/volunteer/EventsList.tsx
 *
 * Events section of the volunteer page.
 * - Organizers see "Switch to Volunteer" banner + "Create Event" button
 * - Volunteers see "Become Organizer" banner
 * - Both roles can join/leave events they don't own
 */

import EventCard from './EventCard';
import type { EventWithMeta } from '../types/volunteer';
import type { Event } from '@/lib/supabase';

type Props = {
  events: EventWithMeta[];
  eventsLoading: boolean;
  currentUser: any;
  isOrganizer: boolean;
  roleLoading: boolean;
  isMyEvent: (ev: Event) => boolean;
  onCreateEvent: () => void;
  onBecomeOrganizer: () => void;
  onBecomeVolunteer: () => void;
  onJoin: (eventId: string) => void;
  onLeave: (eventId: string) => void;
  onEdit: (ev: Event) => void;
  onDelete: (eventId: string) => void;
  onViewDetail: (ev: EventWithMeta) => void;
};

export default function EventsList({
  events, eventsLoading, currentUser, isOrganizer, roleLoading,
  isMyEvent, onCreateEvent, onBecomeOrganizer, onBecomeVolunteer,
  onJoin, onLeave, onEdit, onDelete, onViewDetail,
}: Props) {
  return (
    <section className="py-24 px-4 bg-gray-50" id="events">
      <div className="max-w-6xl mx-auto">

        {/* Section header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-12 gap-4">
          <div>
            <span className="text-green-600 font-semibold text-sm uppercase tracking-wider">Community Drives</span>
            <h2 className="text-3xl font-bold text-gray-900 mt-1">Upcoming Clean-Up Events</h2>
          </div>
          {isOrganizer && (
            <button
              onClick={onCreateEvent}
              className="bg-green-600 text-white px-6 py-3 rounded-full font-semibold hover:bg-green-700 transition-colors flex items-center gap-2"
            >
              <span className="text-lg">+</span> Create Event
            </button>
          )}
        </div>

        {/* Loading */}
        {eventsLoading ? (
          <div className="text-center py-20 text-gray-400">
            <div className="text-4xl mb-4">🌿</div>
            <p>Loading events...</p>
          </div>

        /* Empty state */
        ) : events.length === 0 ? (
          <div className="text-center py-24 border-2 border-dashed border-green-200 rounded-2xl bg-white">
            <div className="text-5xl mb-6">🗂️</div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">No Events Yet</h3>
            <p className="text-gray-500 mb-8 max-w-sm mx-auto">
              There are no clean-up drives scheduled yet. Be the first to organize one for your barangay!
            </p>
            {!currentUser ? (
              <p className="text-sm text-gray-400">Sign in first, then become an organizer to create events.</p>
            ) : !isOrganizer ? (
              <button
                onClick={onBecomeOrganizer}
                disabled={roleLoading}
                className="bg-green-600 text-white px-8 py-3 rounded-full font-semibold hover:bg-green-700 disabled:opacity-60 transition-colors"
              >
                {roleLoading ? 'Switching...' : '🏆 Become an Organizer'}
              </button>
            ) : (
              <button
                onClick={onCreateEvent}
                className="bg-green-600 text-white px-8 py-3 rounded-full font-semibold hover:bg-green-700 transition-colors"
              >
                + Create Your First Event
              </button>
            )}
          </div>

        /* Events list */
        ) : (
          <div className="space-y-4">
            {events.map((ev) => (
              <EventCard
                key={ev.id}
                event={ev}
                currentUser={currentUser}
                isMyEvent={isMyEvent(ev)}
                onJoin={() => onJoin(ev.id)}
                onLeave={() => onLeave(ev.id)}
                onEdit={() => onEdit(ev)}
                onDelete={() => onDelete(ev.id)}
                onViewDetail={() => onViewDetail(ev)}
              />
            ))}
          </div>
        )}

        {/* Role switch banners — shown below the events list when logged in */}
        {currentUser && events.length >= 0 && (
          <div className="mt-12">

            {/* Volunteer → Organizer */}
            {!isOrganizer && (
              <div className="bg-green-700 text-white rounded-2xl p-8 text-center">
                <h3 className="text-xl font-bold mb-2">Want to lead your own clean-up drive?</h3>
                <p className="text-green-200 mb-6 text-sm">
                  Become an Organizer to create and manage events for your community.
                </p>
                <button
                  onClick={onBecomeOrganizer}
                  disabled={roleLoading}
                  className="bg-white text-green-700 font-bold px-8 py-3 rounded-full hover:bg-green-50 disabled:opacity-60 transition-colors"
                >
                  {roleLoading ? 'Switching...' : '🏆 Become an Organizer'}
                </button>
              </div>
            )}

            {/* Organizer → Volunteer */}
            {isOrganizer && (
              <div className="bg-white border-2 border-green-200 rounded-2xl p-8 text-center">
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  Want to join as a volunteer instead?
                </h3>
                <p className="text-gray-500 mb-6 text-sm">
                  Switch to volunteer mode to join events organized by others.
                  You can always become an organizer again later.
                </p>
                <button
                  onClick={onBecomeVolunteer}
                  disabled={roleLoading}
                  className="bg-green-600 text-white font-bold px-8 py-3 rounded-full hover:bg-green-700 disabled:opacity-60 transition-colors"
                >
                  {roleLoading ? 'Switching...' : '🙋 Switch to Volunteer'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}