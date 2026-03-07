import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

/**
 * About Page — DavaoClean AI
 * Covers mission, background on the waste crisis, team profiles,
 * key statistics, and core features aligned with SDG 11 & 12.
 */
export default function AboutPage() {
  return (
    <main className="min-h-screen bg-white">
      <Navbar />
      
      {/* Spacer for fixed navbar */}
      <div className="h-16" />

      {/* ─── Header / Hero Section ─── */}
      <section className="relative bg-linear-to-br from-green-700 to-emerald-800 text-white py-24 px-4 overflow-hidden">
        {/* Decorative background circles */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-white opacity-5 rounded-full translate-x-1/2 -translate-y-1/3" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-white opacity-5 rounded-full -translate-x-1/3 translate-y-1/3" />

        <div className="max-w-4xl mx-auto text-center relative z-10">
          <span className="inline-block bg-green-600 text-green-100 text-sm font-medium px-4 py-1 rounded-full mb-6">
            🌿 Our Story
          </span>
          <h1 className="text-5xl md:text-6xl font-extrabold mb-6 leading-tight">
            About DavaoClean AI
          </h1>
          <p className="text-xl text-green-100 max-w-2xl mx-auto leading-relaxed">
            Our mission is to revolutionize waste management in Davao City by harnessing the power
            of artificial intelligence — making proper segregation accessible, effortless, and impactful
            for every resident.
          </p>
        </div>
      </section>

      {/* ─── Background / About Section ─── */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            {/* Text */}
            <div>
              <span className="text-green-600 font-semibold text-sm uppercase tracking-wider">The Problem</span>
              <h2 className="text-3xl font-bold text-gray-900 mt-2 mb-6">Davao's Growing Landfill Crisis</h2>
              <p className="text-gray-600 leading-relaxed mb-4">
                Davao City generates over 500 tons of waste daily, with the Binugao landfill already
                operating at 85% capacity. Without urgent intervention, the city faces a looming waste
                management crisis that threatens public health, the environment, and local communities.
              </p>
              <p className="text-gray-600 leading-relaxed mb-4">
                A major contributor to the crisis is improper waste segregation. Many residents lack
                the knowledge or tools to correctly separate biodegradable, recyclable, residual, and
                special wastes — leading to contamination, inefficient collection, and increased landfill load.
              </p>
              <p className="text-gray-600 leading-relaxed">
                DavaoClean AI was born from the vision that technology can bridge this gap. By providing
                an instant, AI-powered waste classifier, we empower every Davaoeño to become an active
                participant in solving this crisis — one photo at a time.
              </p>
            </div>

            {/* Visual card */}
            <div className="bg-green-50 border border-green-200 rounded-2xl p-8 space-y-6">
              {[
                { emoji: '♻️', label: 'Proper Segregation', desc: 'Correct waste sorting at the source reduces landfill pressure significantly.' },
                { emoji: '📱', label: 'Digital Solution', desc: 'Mobile-friendly AI tools meet residents where they are — on their phones.' },
                { emoji: '🤝', label: 'Community Action', desc: 'Clean-up drives and volunteer programs amplify individual impact.' },
              ].map((item) => (
                <div key={item.label} className="flex gap-4 items-start">
                  <div className="text-2xl">{item.emoji}</div>
                  <div>
                    <div className="font-semibold text-gray-900">{item.label}</div>
                    <div className="text-gray-600 text-sm">{item.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ─── Team Section ─── */}
      <section className="py-24 px-4 bg-green-50">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <span className="text-green-600 font-semibold text-sm uppercase tracking-wider">The People Behind It</span>
            <h2 className="text-3xl font-bold text-gray-900 mt-2 mb-4">Meet the Team</h2>
            <p className="text-gray-500 max-w-xl mx-auto">
              DavaoClean AI was built by two passionate developers who believe that technology
              and environmental responsibility go hand in hand.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {[
              {
                name: 'Ivann Peria',
                role: 'Full Stack Developer',
                bio: 'Ivann leads the frontend and AI integration, crafting seamless user experiences and connecting the waste classification model to the live application.',
                initials: 'IP',
                color: 'bg-green-600',
              },
              {
                name: 'Alle Zayd Abusaman',
                role: 'Full Stack Developer',
                bio: 'Alle handles the backend architecture, database design with Supabase, and ensures the system is scalable, secure, and ready for city-wide use.',
                initials: 'AZ',
                color: 'bg-emerald-600',
              },
            ].map((member) => (
              <div
                key={member.name}
                className="bg-white border border-green-100 rounded-2xl p-8 text-center hover:shadow-lg transition-shadow"
              >
                {/* Avatar placeholder — replace with actual <Image> when photos are available */}
                <div className={`w-24 h-24 ${member.color} rounded-full flex items-center justify-center mx-auto mb-6`}>
                  <span className="text-white text-3xl font-bold">{member.initials}</span>
                </div>
                <h3 className="text-xl font-bold text-gray-900">{member.name}</h3>
                <p className="text-green-600 font-medium text-sm mb-4">{member.role}</p>
                <p className="text-gray-600 text-sm leading-relaxed">{member.bio}</p>
              </div>
            ))}
          </div>

          <p className="text-center text-gray-500 text-sm mt-8 max-w-xl mx-auto">
            Together, Ivann and Alle Zayd combine design, engineering, and a shared passion for Davao City
            to build tools that create real-world environmental impact.
          </p>
        </div>
      </section>

      {/* ─── Stats Section ─── */}
      <section className="py-20 px-4 bg-green-700 text-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-2">The Numbers Behind the Mission</h2>
            <p className="text-green-200 text-sm">Why DavaoClean AI is urgently needed</p>
          </div>
          <div className="grid md:grid-cols-4 gap-8 text-center">
            {[
              { value: '500+', label: 'Tons of Waste Daily' },
              { value: '85%', label: 'Landfill Capacity Used' },
              { value: '40%', label: 'Projected Diversion Improvement' },
              { value: '95%+', label: 'AI Classification Accuracy' },
            ].map((stat) => (
              <div key={stat.value}>
                <div className="text-5xl font-extrabold mb-2">{stat.value}</div>
                <div className="text-green-200 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Features List Section ─── */}
      <section className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <span className="text-green-600 font-semibold text-sm uppercase tracking-wider">What We Built</span>
            <h2 className="text-3xl font-bold text-gray-900 mt-2 mb-4">Core Features</h2>
            <p className="text-gray-500">
              DavaoClean AI is designed around three pillars that directly address Davao's waste challenges
              while supporting <span className="font-semibold text-green-700">UN SDG 11</span> (Sustainable Cities)
              and <span className="font-semibold text-green-700">SDG 12</span> (Responsible Consumption).
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                emoji: '🤖',
                title: 'AI Waste Classifier',
                desc: 'Upload a photo and instantly know your waste type — Biodegradable, Recyclable, Residual, or Special Waste — with over 95% accuracy. Proper disposal instructions included.',
                badge: 'SDG 12',
              },
              {
                emoji: '📅',
                title: 'Collection Schedule Dashboard',
                desc: 'View barangay-specific waste collection schedules so residents know exactly when and what to put out — eliminating missed pickups and mixed waste.',
                badge: 'SDG 11',
              },
              {
                emoji: '📣',
                title: 'Reporting & Volunteer Tool',
                desc: 'Report illegal dumpsites and join community clean-up drives. Mobilizing citizens as environmental stewards amplifies the impact of technology.',
                badge: 'SDG 11 & 12',
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className="bg-white border border-green-100 rounded-2xl p-8 hover:border-green-400 hover:shadow-md transition-all"
              >
                <div className="text-4xl mb-4">{feature.emoji}</div>
                <div className="inline-block bg-green-100 text-green-700 text-xs font-semibold px-3 py-1 rounded-full mb-4">
                  {feature.badge}
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}