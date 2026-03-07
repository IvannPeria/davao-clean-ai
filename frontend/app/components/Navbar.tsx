'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

/**
 * Navbar Component
 * Shared navigation bar used across all pages.
 * Highlights the active route and includes a mobile hamburger menu.
 */
export default function Navbar() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/about', label: 'About' },
    { href: '/classifier', label: 'Classifier' },
    { href: '/volunteer', label: 'Volunteer' },
  ];

  return (
    <nav className="fixed top-0 w-full bg-white border-b border-green-100 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <span className="text-2xl">🌿</span>
            <span className="text-2xl font-bold text-green-800 tracking-tight">
              DavaoClean <span className="text-green-500">AI</span>
            </span>
          </Link>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm font-medium transition-colors ${
                  pathname === link.href
                    ? 'text-green-600 border-b-2 border-green-500 pb-0.5'
                    : 'text-gray-600 hover:text-green-600'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* CTA Button */}
          <Link
            href="/classifier"
            className="hidden md:inline-block bg-green-600 text-white px-5 py-2 rounded-full text-sm font-semibold hover:bg-green-700 transition-colors"
          >
            Try Classifier
          </Link>

          {/* Mobile Hamburger */}
          <button
            className="md:hidden text-gray-600 hover:text-green-600 focus:outline-none"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {menuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Menu Dropdown */}
        {menuOpen && (
          <div className="md:hidden pb-4 pt-2 space-y-2 border-t border-green-100 mt-2">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className={`block px-2 py-2 rounded-md text-sm font-medium ${
                  pathname === link.href
                    ? 'text-green-600 bg-green-50'
                    : 'text-gray-600 hover:text-green-600 hover:bg-green-50'
                }`}
              >
                {link.label}
              </Link>
            ))}
            <Link
              href="/classifier"
              onClick={() => setMenuOpen(false)}
              className="block w-full text-center mt-2 bg-green-600 text-white px-5 py-2 rounded-full text-sm font-semibold hover:bg-green-700 transition-colors"
            >
              Try Classifier
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}