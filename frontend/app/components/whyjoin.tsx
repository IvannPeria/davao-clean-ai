/**
 * components/volunteer/WhyJoin.tsx
 *
 * Static "Why Volunteer?" awareness section.
 * No props needed — content is fixed.
 */

const REASONS = [
  { emoji: '🌱', title: 'Real Environmental Impact', desc: 'Each drive prevents hundreds of kilograms of waste from entering landfills, rivers, and barangay streets.' },
  { emoji: '🤝', title: 'Build Community Bonds', desc: 'Clean-up drives bring Davaoeños together — neighbors, students, and businesses working as one.' },
  { emoji: '📸', title: 'Document & Share', desc: 'Upload photos from each drive to build a public record of community action and inspire others to join.' },
  { emoji: '🏆', title: 'Lead as an Organizer', desc: 'Create and manage your own clean-up events, set locations, and mobilize volunteers in your barangay.' },
  { emoji: '📅', title: 'Flexible Commitment', desc: 'Join single events or commit to recurring drives — participation is entirely on your own schedule.' },
  { emoji: '🌐', title: 'SDG Aligned Action', desc: 'Your involvement directly supports UN Sustainable Development Goals 11 (Cities) and 12 (Consumption).' },
];

export default function WhyJoin() {
  return (
    <section className="py-24 px-4 bg-white">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <span className="text-green-600 font-semibold text-sm uppercase tracking-wider">Why Volunteer?</span>
          <h2 className="text-3xl font-bold text-gray-900 mt-2 mb-4">Be the Change Davao Needs</h2>
          <p className="text-gray-500 max-w-xl mx-auto">
            Every clean-up drive removes waste before it reaches the landfill.
            Your participation directly contributes to SDG 11 and SDG 12 goals.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {REASONS.map((item) => (
            <div
              key={item.title}
              className="bg-green-50 border border-green-100 rounded-2xl p-6 hover:border-green-300 hover:shadow-md transition-all"
            >
              <div className="text-3xl mb-3">{item.emoji}</div>
              <h3 className="font-bold text-gray-900 mb-2">{item.title}</h3>
              <p className="text-gray-600 text-sm leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}