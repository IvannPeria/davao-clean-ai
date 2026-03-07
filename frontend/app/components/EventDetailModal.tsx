/**
 * components/volunteer/EventDetailModal.tsx
 *
 * Modal showing full event details: description, participants list,
 * and a photo gallery with uploader attribution.
 *
 * - Organizers can accept, decline, or remove participants.
 * - Accepted participants and the organizer can upload photos.
 * - Every photo shows the uploader's email + date — always visible, not just on hover.
 * - Uploaders can delete their own photos.
 */

import type { Profile, Upload } from '@/lib/supabase';
import type { EventWithMeta, ParticipantWithEmail, UploadWithEmail } from '../types/volunteer';

type Props = {
  event: EventWithMeta;
  participants: ParticipantWithEmail[];
  uploads: UploadWithEmail[];
  currentUser: any;
  profile: Profile | null;
  detailLoading: boolean;
  uploadLoading: boolean;
  uploadError: string;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onClose: () => void;
  onUpdateParticipantStatus: (id: string, status: 'accepted' | 'declined') => void;
  onRemoveParticipant: (id: string) => void;
  onDeleteUpload: (upload: Upload) => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onUploadClick: () => void;
};

export default function EventDetailModal({
  event, participants, uploads, currentUser, profile,
  detailLoading, uploadLoading, uploadError, fileInputRef,
  onClose, onUpdateParticipantStatus, onRemoveParticipant,
  onDeleteUpload, onFileChange, onUploadClick,
}: Props) {
  const isMyEvent = currentUser && event.organizer_id === currentUser.id;
  const isParticipant = participants.some(
    (p) => p.volunteer_id === currentUser?.id && p.status === 'accepted'
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50">
      <div className="bg-white rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">

        {/* Sticky modal header */}
        <div className="sticky top-0 bg-white border-b border-gray-100 p-6 flex items-start justify-between rounded-t-2xl z-10">
          <div>
            <h2 className="text-xl font-bold text-gray-900">{event.title}</h2>
            <p className="text-sm text-gray-500">
              📍 {event.location} · 📅 {new Date(event.event_date).toLocaleDateString('en-PH', { dateStyle: 'medium' })}
            </p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 ml-4 mt-1">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 space-y-8">
          {detailLoading ? (
            <div className="text-center py-12 text-gray-400">
              <div className="text-3xl mb-2">🌿</div>
              <p>Loading details...</p>
            </div>
          ) : (
            <>
              {/* Event description */}
              {event.description && (
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">About this Event</h3>
                  <p className="text-gray-600 text-sm leading-relaxed">{event.description}</p>
                </div>
              )}

              {/* ── Participants list ── */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">
                  Participants ({participants.length})
                </h3>
                {participants.length === 0 ? (
                  <p className="text-gray-400 text-sm italic">No participants yet.</p>
                ) : (
                  <div className="space-y-2">
                    {participants.map((p) => (
                      <div
                        key={p.id}
                        className="flex items-center justify-between bg-gray-50 rounded-xl px-4 py-3"
                      >
                        <div className="flex items-center gap-3">
                          {/* Avatar initial */}
                          <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-700 font-bold text-xs">
                            {p.email?.[0]?.toUpperCase()}
                          </div>
                          <div>
                            <div className="text-sm font-medium text-gray-900">{p.email}</div>
                            <div className={`text-xs font-medium ${
                              p.status === 'accepted' ? 'text-green-600'
                              : p.status === 'declined' ? 'text-red-500'
                              : 'text-yellow-600'
                            }`}>
                              {p.status}
                            </div>
                          </div>
                        </div>

                        {/* Organizer participant controls */}
                        {isMyEvent && (
                          <div className="flex gap-2">
                            {p.status !== 'accepted' && (
                              <button
                                onClick={() => onUpdateParticipantStatus(p.id, 'accepted')}
                                className="text-xs text-green-600 border border-green-200 px-3 py-1 rounded-full hover:bg-green-50 transition-colors"
                              >
                                Accept
                              </button>
                            )}
                            {p.status !== 'declined' && (
                              <button
                                onClick={() => onUpdateParticipantStatus(p.id, 'declined')}
                                className="text-xs text-red-600 border border-red-200 px-3 py-1 rounded-full hover:bg-red-50 transition-colors"
                              >
                                Decline
                              </button>
                            )}
                            <button
                              onClick={() => onRemoveParticipant(p.id)}
                              className="text-xs text-gray-500 border border-gray-200 px-3 py-1 rounded-full hover:bg-gray-100 transition-colors"
                            >
                              Remove
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* ── Photo gallery ── */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-gray-900">
                    Event Photos ({uploads.length})
                  </h3>

                  {/* Upload button — accepted participants and organizer only */}
                  {currentUser && (isParticipant || isMyEvent) && (
                    <button
                      onClick={onUploadClick}
                      disabled={uploadLoading}
                      className="text-sm bg-green-600 text-white px-4 py-2 rounded-full hover:bg-green-700 disabled:opacity-60 transition-colors flex items-center gap-1.5"
                    >
                      {uploadLoading ? (
                        <>
                          <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                          </svg>
                          Uploading...
                        </>
                      ) : '📷 Upload Photo'}
                    </button>
                  )}

                  {/* Hidden file input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    onChange={onFileChange}
                    className="hidden"
                  />
                </div>

                {/* Upload error */}
                {uploadError && (
                  <div className="bg-red-50 border border-red-200 text-red-700 text-xs px-4 py-2 rounded-xl mb-3">
                    {uploadError}
                  </div>
                )}

                {/* Access hints */}
                {!currentUser && (
                  <p className="text-gray-400 text-xs mb-3">
                    Sign in and join this event to upload photos.
                  </p>
                )}
                {currentUser && !isParticipant && !isMyEvent && (
                  <p className="text-gray-400 text-xs mb-3">
                    Join this event to upload photos.
                  </p>
                )}

                {/* Empty state */}
                {uploads.length === 0 ? (
                  <div className="text-center py-10 border-2 border-dashed border-gray-200 rounded-xl text-gray-400 text-sm">
                    <div className="text-3xl mb-2">📷</div>
                    No photos yet. Be the first to upload!
                  </div>
                ) : (
                  /* Photo grid — each card shows photo + uploader info below */
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {uploads.map((u) => (
                      <div
                        key={u.id}
                        className="group rounded-xl overflow-hidden bg-gray-50 border border-gray-200 hover:border-green-300 hover:shadow-md transition-all"
                      >
                        {/* Photo */}
                        <div className="relative aspect-square bg-gray-100">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={u.image_url}
                            alt={`Photo by ${u.email}`}
                            className="w-full h-full object-cover"
                          />
                          {/* Delete — only the uploader can delete, shown on hover */}
                          {currentUser && u.user_id === currentUser.id && (
                            <button
                              onClick={() => onDeleteUpload(u)}
                              title="Delete your photo"
                              className="absolute top-2 right-2 bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-700"
                            >
                              ✕
                            </button>
                          )}
                        </div>

                        {/* Uploader info — always visible, not hidden on hover */}
                        <div className="px-2.5 py-2 flex items-center gap-2 bg-white">
                          <div className="w-5 h-5 rounded-full bg-green-100 text-green-700 flex items-center justify-center text-xs font-bold shrink-0">
                            {u.email?.[0]?.toUpperCase() ?? '?'}
                          </div>
                          <div className="min-w-0 flex-1">
                            <p
                              className="text-xs font-medium text-gray-800 truncate leading-tight"
                              title={u.email}
                            >
                              {u.email ?? 'Unknown'}
                            </p>
                            <p className="text-xs text-gray-400 leading-tight">
                              {new Date(u.created_at).toLocaleDateString('en-PH', {
                                month: 'short', day: 'numeric',
                              })}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}