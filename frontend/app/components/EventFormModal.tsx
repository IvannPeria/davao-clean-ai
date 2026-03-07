/**
 * components/volunteer/EventFormModal.tsx
 *
 * Modal form for creating or editing a clean-up event.
 * Used exclusively by organizers.
 */

import type { EventModalMode, EventFormState } from '../types/volunteer';

type Props = {
  mode: EventModalMode;
  form: EventFormState;
  formError: string;
  formLoading: boolean;
  onChange: (field: keyof EventFormState, val: string) => void;
  onSubmit: () => void;
  onClose: () => void;
};

export default function EventFormModal({
  mode, form, formError, formLoading,
  onChange, onSubmit, onClose,
}: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50">
      <div className="bg-white rounded-2xl w-full max-w-md shadow-2xl">

        {/* Modal header */}
        <div className="p-6 border-b border-gray-100 flex items-center justify-between">
          <h2 className="text-xl font-bold text-gray-900">
            {mode === 'create' ? 'Create New Event' : 'Edit Event'}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form fields */}
        <div className="p-6 space-y-4">
          {formError && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-xl">
              {formError}
            </div>
          )}

          {/* Text fields: title and location */}
          {[
            { field: 'title' as const, label: 'Event Title *', placeholder: 'e.g. Barangay 76-A Coastal Clean-up' },
            { field: 'location' as const, label: 'Location *', placeholder: 'e.g. Bucana Beach, Barangay 76-A' },
          ].map(({ field, label, placeholder }) => (
            <div key={field}>
              <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
              <input
                type="text"
                placeholder={placeholder}
                value={form[field]}
                onChange={(e) => onChange(field, e.target.value)}
                className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
              />
            </div>
          ))}

          {/* Description textarea */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              placeholder="Describe the event, what to bring, dress code, etc."
              value={form.description}
              onChange={(e) => onChange('description', e.target.value)}
              rows={3}
              className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400 resize-none text-gray-900 placeholder-gray-400 bg-white"
            />
          </div>

          {/* Date/time picker */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Event Date & Time *</label>
            <input
              type="datetime-local"
              value={form.event_date}
              onChange={(e) => onChange('event_date', e.target.value)}
              className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-900 placeholder-gray-400 bg-white focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
            />
          </div>
        </div>

        {/* Footer actions */}
        <div className="p-6 pt-0 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 border border-gray-200 text-gray-700 py-3 rounded-xl font-medium hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onSubmit}
            disabled={formLoading}
            className="flex-1 bg-green-600 text-white py-3 rounded-xl font-semibold hover:bg-green-700 disabled:opacity-60 transition-colors"
          >
            {formLoading ? 'Saving...' : mode === 'create' ? 'Create Event' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
}