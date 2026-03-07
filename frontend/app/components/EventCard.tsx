/**
 * components/volunteer/EventCard.tsx
 *
 * Single event row. Join/Leave is available to ALL logged-in users
 * on events they don't own — including organizers joining other events.
 */

import type { EventWithMeta } from '../types/volunteer';

type Props = {
  event: EventWithMeta;
  currentUser: any;
  isMyEvent: boolean;
  onJoin: () => void;
  onLeave: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onViewDetail: () => void;
};

export default function EventCard({
  event, currentUser, isMyEvent,
  onJoin, onLeave, onEdit, onDelete, onViewDetail,
}: Props) {
  const eventDate = new Date(event.event_date);
  const isPast    = eventDate < new Date();
  const hasJoined = !!event.user_participation;

  return (
    <div className={`bg-white border rounded-2xl p-6 hover:shadow-md transition-all ${
      isPast ? 'border-gray-200 opacity-75' : 'border-green-100 hover:border-green-300'
    }`}>
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">

        {/* Event info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-1">
            <h3 className="text-lg font-bold text-gray-900 truncate">{event.title}</h3>
            {isPast && (
              <span className="shrink-0 bg-gray-100 text-gray-500 text-xs px-2 py-0.5 rounded-full">
                Past
              </span>
            )}
            {isMyEvent && (
              <span className="shrink-0 bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full font-medium">
                Your Event
              </span>
            )}
            {!isMyEvent && hasJoined && (
              <span className="shrink-0 bg-blue-100 text-blue-700 text-xs px-2 py-0.5 rounded-full font-medium">
                Joined
              </span>
            )}
          </div>
          <p className="text-gray-500 text-sm mb-2 line-clamp-1">{event.description}</p>
          <div className="flex flex-wrap gap-4 text-xs text-gray-500">
            <span>📍 {event.location}</span>
            <span>📅 {eventDate.toLocaleDateString('en-PH', {
              weekday: 'short', year: 'numeric', month: 'short',
              day: 'numeric', hour: '2-digit', minute: '2-digit',
            })}</span>
            <span>👥 {event.participant_count} joined</span>
            <span>🧑‍💼 {event.organizer_email}</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onViewDetail}
            className="text-sm text-green-600 border border-green-200 px-4 py-2 rounded-full hover:bg-green-50 transition-colors"
          >
            View Details
          </button>

          {/*
            Join / Leave — shown to ANY logged-in user on events they don't own,
            including organizers who want to participate in someone else's event.
            Hidden for past events and the organizer's own events.
          */}
          {currentUser && !isMyEvent && !isPast && (
            hasJoined ? (
              <button
                onClick={onLeave}
                className="text-sm text-red-600 border border-red-200 px-4 py-2 rounded-full hover:bg-red-50 transition-colors"
              >
                Leave
              </button>
            ) : (
              <button
                onClick={onJoin}
                className="text-sm bg-green-600 text-white px-4 py-2 rounded-full hover:bg-green-700 transition-colors"
              >
                Join
              </button>
            )
          )}

          {/* Edit / Delete — organizer's own events only */}
          {isMyEvent && (
            <>
              <button
                onClick={onEdit}
                className="text-sm text-gray-600 border border-gray-200 px-4 py-2 rounded-full hover:bg-gray-50 transition-colors"
              >
                Edit
              </button>
              <button
                onClick={onDelete}
                className="text-sm text-red-600 border border-red-200 px-4 py-2 rounded-full hover:bg-red-50 transition-colors"
              >
                Delete
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}