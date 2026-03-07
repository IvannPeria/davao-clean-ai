'use client';

/**
 * app/volunteer/page.tsx
 *
 * Volunteer & Organizer page — orchestrator only.
 * All data/logic lives in hooks/useVolunteer.ts.
 * All UI sections live in components/volunteer/.
 */

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import { useVolunteer } from '../hooks/useVolunteer';

import VolunteerHeader  from '../components/VolunteerHeader';
import WhyJoin          from '../components/whyjoin';
import EventsList       from '../components/EventsList';
import AuthSection      from '../components/AuthSection';
import EventDetailModal from '../components/EventDetailModal';
import EventFormModal   from '../components/EventFormModal';

export default function VolunteerPage() {
  const v = useVolunteer();

  return (
    <main className="min-h-screen bg-white">
      <Navbar />
      <div className="h-16" />

      <VolunteerHeader
        currentUser={v.currentUser}
        profile={v.profile}
        authLoading={v.authLoading}
        onSignUpClick={() => {
          v.setAuthMode('signup');
          document.getElementById('auth-section')?.scrollIntoView({ behavior: 'smooth' });
        }}
        onSignInClick={() => {
          v.setAuthMode('signin');
          document.getElementById('auth-section')?.scrollIntoView({ behavior: 'smooth' });
        }}
        onSignOut={v.handleSignOut}
      />

      <WhyJoin />

      <EventsList
        events={v.events}
        eventsLoading={v.eventsLoading}
        currentUser={v.currentUser}
        isOrganizer={v.isOrganizer}
        roleLoading={v.roleLoading}
        isMyEvent={v.isMyEvent}
        onCreateEvent={v.openCreateModal}
        onBecomeOrganizer={v.handleBecomeOrganizer}
        onBecomeVolunteer={v.handleBecomeVolunteer}
        onJoin={v.handleJoinEvent}
        onLeave={v.handleLeaveEvent}
        onEdit={v.openEditModal}
        onDelete={v.handleDeleteEvent}
        onViewDetail={v.fetchEventDetail}
      />

      {/* Auth form — only shown to guests */}
      {!v.currentUser && !v.authLoading && (
        <AuthSection
          authMode={v.authMode}
          authEmail={v.authEmail}
          authPassword={v.authPassword}
          authError={v.authError}
          authFormLoading={v.authFormLoading}
          onEmailChange={v.setAuthEmail}
          onPasswordChange={v.setAuthPassword}
          onSignIn={v.handleSignIn}
          onSignUp={v.handleSignUp}
          onToggleMode={() => {
            v.setAuthMode(v.authMode === 'signin' ? 'signup' : 'signin');
          }}
        />
      )}

      {/* Event detail modal */}
      {v.selectedEvent && (
        <EventDetailModal
          event={v.selectedEvent}
          participants={v.eventParticipants}
          uploads={v.eventUploads}
          currentUser={v.currentUser}
          profile={v.profile}
          detailLoading={v.detailLoading}
          uploadLoading={v.uploadLoading}
          uploadError={v.uploadError}
          fileInputRef={v.fileInputRef}
          onClose={() => { v.setSelectedEvent(null); v.setUploadError(''); }}
          onUpdateParticipantStatus={v.handleUpdateParticipantStatus}
          onRemoveParticipant={v.handleRemoveParticipant}
          onDeleteUpload={v.handleDeleteUpload}
          onFileChange={(e) => { const f = e.target.files?.[0]; if (f) v.handleImageUpload(f); }}
          onUploadClick={() => v.fileInputRef.current?.click()}
        />
      )}

      {/* Create / edit event modal */}
      {v.eventModalMode && (
        <EventFormModal
          mode={v.eventModalMode}
          form={v.eventForm}
          formError={v.formError}
          formLoading={v.formLoading}
          onChange={(field, val) => v.setEventForm((prev) => ({ ...prev, [field]: val }))}
          onSubmit={v.handleEventFormSubmit}
          onClose={() => v.setEventModalMode(null)}
        />
      )}

      <Footer />
    </main>
  );
}