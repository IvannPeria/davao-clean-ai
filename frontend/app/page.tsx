import Link from 'next/link';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

/**
 * Home Page — DavaoClean AI
 * Landing page with hero, how it works, stats, benefits, FAQ, and CTA sections.
 * Green-themed to reflect environmental mission.
 */
export default function Home() {
  return (
    <main className="min-h-screen bg-white">
      <Navbar />

      {/* Spacer for fixed navbar */}
      <div className="h-16" />

      {/* ─── Hero Section ─── */}
      <section className="relative overflow-hidden bg-linear-to-b from-green-50 to-white pt-12 pb-24 px-4">
        {/* Decorative blobs */}
        <div className="absolute top-0 left-0 w-96 h-96 bg-green-200 rounded-full opacity-20 -translate-x-1/2 -translate-y-1/2 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-80 h-80 bg-emerald-300 rounded-full opacity-20 translate-x-1/3 translate-y-1/3 blur-3xl pointer-events-none" />

        <div className="max-w-5xl mx-auto text-center relative z-10">
          {/* Badge */}
          <span className="inline-block bg-green-100 text-green-700 text-sm font-medium px-4 py-1 rounded-full mb-6">
            🌱 AI-Powered Waste Management
          </span>

          <h1 className="text-5xl md:text-6xl font-extrabold text-gray-900 leading-tight mb-6">
            Smart Waste Segregation
            <br />
            <span className="text-green-600">for Davao City</span>
          </h1>

          <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
            DavaoClean AI revolutionizes waste management with AI-powered photo classification.
            Upload a waste photo, get instant results, and view collection schedules—free for all Davaoeños.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/classifier"
              className="bg-green-600 text-white px-8 py-3 rounded-full text-lg font-semibold hover:bg-green-700 transition-colors shadow-lg shadow-green-200"
            >
              Classify Waste Now
            </Link>
            <Link
              href="/about"
              className="border-2 border-green-300 text-green-700 px-8 py-3 rounded-full text-lg font-semibold hover:bg-green-50 transition-colors"
            >
              Learn More
            </Link>
          </div>
        </div>

        {/* Hero Image */}
        <div className="max-w-5xl mx-auto mt-14">
          <img
            src="/img/main1.png"
            alt="DavaoClean AI waste management illustration"
            className="w-full h-auto rounded-2xl shadow-xl"
          />
        </div>
      </section>

      {/* ─── How It Works ─── */}
      <section className="py-24 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-gray-500">Three simple steps to properly segregate your waste</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: '01',
                title: 'Upload a Photo',
                desc: 'Take or upload a clear photo of your waste item using your phone or device camera.',
                icon: (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                ),
              },
              {
                step: '02',
                title: 'AI Classification',
                desc: 'Our AI model instantly analyzes and categorizes your waste into the correct type.',
                icon: (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                ),
              },
              {
                step: '03',
                title: 'View Schedule',
                desc: 'Get proper disposal instructions and the collection schedule for your barangay.',
                icon: (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                ),
              },
            ].map((item) => (
              <div key={item.step} className="relative bg-white border border-green-100 rounded-2xl p-8 hover:shadow-lg hover:border-green-300 transition-all group">
                <div className="text-5xl font-black text-green-100 group-hover:text-green-200 transition-colors mb-4">{item.step}</div>
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-4">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    {item.icon}
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{item.title}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Stats Section ─── */}
      <section className="py-20 px-4 bg-green-700 text-white">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            {[
              { value: '500+', label: 'Tons of Daily Waste in Davao' },
              { value: '85%', label: 'Current Landfill Capacity' },
              { value: '40%', label: 'Projected Diversion Rate Improvement' },
            ].map((stat) => (
              <div key={stat.value}>
                <div className="text-6xl font-extrabold text-white mb-2">{stat.value}</div>
                <div className="text-green-200 text-sm font-medium">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Benefits / Features Scroll ─── */}
      <section className="py-24 px-4 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Why DavaoClean AI?</h2>
            <p className="text-gray-500">Built for Davao, aligned with global sustainability goals</p>
          </div>

          {/* Horizontally scrollable benefit cards */}
          <div className="flex overflow-x-auto gap-6 pb-6 snap-x snap-mandatory scrollbar-hide">
            {[
              { color: 'green', emoji: '🌍', title: 'Environmental Impact', desc: 'Reduce landfill waste and promote sustainable waste practices across Davao City.' },
              { color: 'emerald', emoji: '✅', title: 'Compliance Made Easy', desc: 'Automatically follow DENR and local waste segregation regulations without guessing.' },
              { color: 'teal', emoji: '🏘️', title: 'Cleaner Neighborhoods', desc: 'Contribute to healthier, more livable barangays for every resident.' },
              { color: 'lime', emoji: '🤖', title: 'Smart AI Technology', desc: 'Powered by advanced image recognition for accurate, real-time waste classification.' },
              { color: 'green', emoji: '⏱️', title: 'Real-time Updates', desc: 'Instant feedback on waste type and up-to-date collection schedule info.' },
              { color: 'emerald', emoji: '🌐', title: 'SDG 11 & 12 Aligned', desc: 'Supporting UN Sustainable Development Goals for cities and responsible consumption.' },
            ].map((item) => (
              <div
                key={item.title}
                className="min-w-70 md:min-w-[320px] bg-white border border-green-100 rounded-2xl p-8 snap-center shrink-0 hover:shadow-md transition-shadow"
              >
                <div className="text-4xl mb-4">{item.emoji}</div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{item.title}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── CTA Section ─── */}
      <section className="py-24 px-4 bg-linear-to-br from-green-700 to-emerald-800 text-white">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-4">Ready to Make a Difference?</h2>
          <p className="text-green-100 text-xl mb-8">
            Start classifying your waste today and help build a cleaner, greener Davao City.
          </p>
          <Link
            href="/classifier"
            className="inline-block bg-white text-green-700 font-bold px-10 py-4 rounded-full text-lg hover:bg-green-50 transition-colors shadow-xl"
          >
            Try the Classifier Free →
          </Link>
        </div>
      </section>

      {/* ─── FAQ Section ─── */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Frequently Asked Questions</h2>
            <p className="text-gray-500">Everything you need to know about DavaoClean AI</p>
          </div>

          <div className="space-y-4">
            {[
              {
                q: 'How accurate is the AI waste classifier?',
                a: 'Our AI has been trained on thousands of waste images and achieves over 95% accuracy across Biodegradable, Recyclable, Residual, and Special Waste categories.',
              },
              {
                q: 'What types of waste can it identify?',
                a: 'Biodegradable (food, garden waste), Recyclable (plastics, paper, metal, glass), Residual (non-recyclable items), and Special Waste (electronics, batteries, hazardous materials).',
              },
              {
                q: 'Are my photos kept private?',
                a: 'Yes. Photos are processed securely and not stored permanently. Only anonymous classification data is retained to improve the model.',
              },
              {
                q: 'How do I use the classifier?',
                a: 'Take or upload a clear photo of your waste item. The AI analyzes it in seconds and returns the waste category plus proper disposal instructions.',
              },
              {
                q: 'What are the collection schedules?',
                a: 'Schedules vary by barangay. After classification, you will see schedule details based on your location and waste type.',
              },
              {
                q: 'Is DavaoClean AI free?',
                a: 'Yes! Completely free for all Davao City residents and businesses as part of our sustainable waste management initiative.',
              },
            ].map((faq, i) => (
              <div key={i} className="border border-green-600 rounded-xl p-6 hover:border-red-500 hover:bg-green-50 transition-all">
                <h3 className="font-semibold text-gray-900 mb-2">{faq.q}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{faq.a}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Contact Section ─── */}
      <section className="py-20 px-4 bg-green-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Need Help?</h2>
            <p className="text-gray-500">Reach out for support or partnership opportunities</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: '✉️',
                title: 'Email',
                sub: 'Send us a message',
                link: 'mailto:support@davaoclean.ai',
                label: 'support@davaoclean.ai',
              },
              {
                icon: '📞',
                title: 'Phone',
                sub: 'Call during business hours',
                link: 'tel:+6382123456',
                label: '+63 (82) 123-4567',
              },
              {
                icon: '📍',
                title: 'Office',
                sub: 'Davao City, Mindanao, Philippines',
                link: '#',
                label: 'Get directions →',
              },
            ].map((c) => (
              <div key={c.title} className="bg-white p-8 rounded-2xl text-center border border-green-100 hover:shadow-md transition-shadow">
                <div className="text-3xl mb-4">{c.icon}</div>
                <h3 className="font-semibold text-gray-900 mb-1">{c.title}</h3>
                <p className="text-sm text-gray-500 mb-2">{c.sub}</p>
                <a href={c.link} className="text-green-600 hover:underline text-sm font-medium">
                  {c.label}
                </a>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}