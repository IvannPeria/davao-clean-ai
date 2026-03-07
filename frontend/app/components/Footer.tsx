import Link from 'next/link';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFacebook } from '@fortawesome/free-brands-svg-icons';

/**
 * Footer Component
 * Shared footer used across all pages.
 * Mobile: brand full-width on top, then a clean 2-column link grid.
 * Desktop: single 5-column row.
 */
export default function Footer() {
  const quickLinks = [
    { href: '/', label: 'Home' },
    { href: '/about', label: 'About' },
    { href: '/classifier', label: 'Classifier' },
    { href: '/volunteer', label: 'Volunteer' },
  ];

  const resources = ['Learn', 'Community', 'Support'];
  const legal = ['Privacy Policy', 'Terms of Service'];

  return (
    <footer className="bg-green-950 text-white">
      <div className="max-w-7xl mx-auto px-6 pt-12 pb-8">

        {/* ── MOBILE layout (hidden on md+) ── */}
        <div className="md:hidden">

          {/* Brand row */}
          <div className="flex items-start gap-3 mb-8 pb-8 border-b border-green-900">
            <span className="text-2xl mt-0.5">🌿</span>
            <div>
              <p className="font-bold text-base tracking-tight mb-1">DavaoClean AI</p>
              <p className="text-green-400 text-sm leading-relaxed">
                Smart waste management for a sustainable Davao City.
                Powered by AI, driven by community.
              </p>
            </div>
          </div>

          {/* 2-column link grid */}
          <div className="grid grid-cols-2 gap-x-8 gap-y-8 mb-8">
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-3">
                Quick Links
              </h4>
              <ul className="space-y-2.5">
                {quickLinks.map((link) => (
                  <li key={link.href}>
                    <Link href={link.href} className="text-green-400 hover:text-white text-sm transition-colors">
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-3">
                Resources
              </h4>
              <ul className="space-y-2.5">
                {resources.map((item) => (
                  <li key={item}>
                    <Link href="#" className="text-green-400 hover:text-white text-sm transition-colors">
                      {item}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-3">
                Legal
              </h4>
              <ul className="space-y-2.5">
                {legal.map((item) => (
                  <li key={item}>
                    <Link href="#" className="text-green-400 hover:text-white text-sm transition-colors">
                      {item}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-3">
                Contact
              </h4>
              <ul className="space-y-2.5">
                <li>
                  <a
                    href="https://facebook.com/allezayd1025"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-green-400 hover:text-white text-sm transition-colors"
                  >
                    <FontAwesomeIcon icon={faFacebook} className="w-3.5 h-3.5 shrink-0" />
                    Facebook
                  </a>
                </li>
                <li>
                  <a
                    href="mailto:azAbusaman@mcm.edu.ph"
                    className="flex items-center gap-2 text-green-400 hover:text-white text-sm transition-colors"
                  >
                    <span className="text-sm">✉️</span>
                    Email
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* ── DESKTOP layout (hidden below md) ── */}
        <div className="hidden md:grid md:grid-cols-5 gap-8 mb-8">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xl">🌿</span>
              <span className="text-lg font-bold tracking-tight">DavaoClean AI</span>
            </div>
            <p className="text-green-400 text-sm leading-relaxed">
              Smart waste management for a sustainable Davao City.
              Powered by AI, driven by community.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-4">Quick Links</h4>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="text-green-400 hover:text-white text-sm transition-colors">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-4">Resources</h4>
            <ul className="space-y-2">
              {resources.map((item) => (
                <li key={item}>
                  <Link href="#" className="text-green-400 hover:text-white text-sm transition-colors">{item}</Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-4">Legal</h4>
            <ul className="space-y-2">
              {legal.map((item) => (
                <li key={item}>
                  <Link href="#" className="text-green-400 hover:text-white text-sm transition-colors">{item}</Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-green-300 mb-4">Contact</h4>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://facebook.com/allezayd1025"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-green-400 hover:text-white text-sm transition-colors"
                >
                  <FontAwesomeIcon icon={faFacebook} className="w-4 h-4 shrink-0" />
                  Facebook
                </a>
              </li>
              <li>
                <a
                  href="mailto:azAbusaman@mcm.edu.ph"
                  className="flex items-center gap-2 text-green-400 hover:text-white text-sm transition-colors"
                >
                  <span className="text-sm">✉️</span>
                  Email
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* ── Bottom bar ── */}
        <div className="border-t border-green-900 pt-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className="text-green-600 text-xs">&copy; {new Date().getFullYear()} DavaoClean AI. All rights reserved.</p>
          <p className="text-green-800 text-xs">Built for Davao City 🌿</p>
        </div>
      </div>
    </footer>
  );
}