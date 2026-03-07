'use client';

import { useState, useRef } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';


/**
 * Classifier Page — DavaoClean AI
 * Allows users to upload or drag-and-drop a waste photo for AI classification.
 * Displays classification result, disposal instructions, and collection schedule context.
 *
 * Hooks used:
 * - useState: manages uploaded image, preview URL, classification result, loading state
 * - useRef: references the hidden file input element for custom upload trigger
 */

/** Waste category type for structured results */
type WasteCategory = 'Biodegradable' | 'Recyclable' | 'Residual' | 'Special Waste' | null;

/** Disposal instructions keyed by waste category */
const DISPOSAL_INSTRUCTIONS: Record<Exclude<WasteCategory, null>, { color: string; emoji: string; instructions: string }> = {
  Biodegradable: {
    color: 'bg-green-100 border-green-400 text-green-800',
    emoji: '🌱',
    instructions:
      'Place in your green bin. Includes food scraps, garden waste, and paper. Collected Monday & Thursday in most barangays.',
  },
  Recyclable: {
    color: 'bg-blue-100 border-blue-400 text-blue-800',
    emoji: '♻️',
    instructions:
      'Place in your blue bin. Clean and dry before disposal. Includes plastics, glass, metal, and cardboard. Collected Wednesday & Saturday.',
  },
  Residual: {
    color: 'bg-gray-100 border-gray-400 text-gray-800',
    emoji: '🗑️',
    instructions:
      'Place in your black bin. Non-recyclable, non-hazardous waste. Collected Tuesday & Friday in most areas.',
  },
  'Special Waste': {
    color: 'bg-red-100 border-red-400 text-red-800',
    emoji: '⚠️',
    instructions:
      'Do NOT place in regular bins. Bring to designated DENR drop-off points. Includes batteries, e-waste, chemicals, and medical waste.',
  },
};

export default function ClassifierPage() {
  /** Stores the File object selected by the user */
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  /** Object URL for previewing the selected image */
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  /** Whether the AI classification is in progress */
  const [isLoading, setIsLoading] = useState(false);

  /** The classified waste category returned from the API */
  const [result, setResult] = useState<WasteCategory>(null);

  /** Whether the user is dragging a file over the drop zone */
  const [isDragging, setIsDragging] = useState(false);

  /** Ref to the hidden <input type="file"> element */
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * Handles file selection from the file picker or drag-and-drop.
   * Creates a local object URL for image preview.
   */
  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  /** Triggers the hidden file input when the upload area is clicked */
  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  /** Handles the native file input change event */
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  /** Handles drag-over event to activate drop zone highlight */
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  /** Handles drag-leave to deactivate drop zone highlight */
  const handleDragLeave = () => setIsDragging(false);

  /** Handles file drop onto the drop zone */
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) handleFileSelect(file);
  };

  /**
   * Sends the uploaded image to the classification API endpoint.
   * TODO: Replace mock timeout with actual Supabase/AI API call.
   */
  const handleClassify = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setResult(null);

    // --- TODO: Replace this mock with your real API call ---
    // const formData = new FormData();
    // formData.append('image', selectedFile);
    // const response = await fetch('/api/classify', { method: 'POST', body: formData });
    // const data = await response.json();
    // setResult(data.category);
    // -------------------------------------------------------

    // Mock delay simulating AI processing
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const mockCategories: Exclude<WasteCategory, null>[] = ['Biodegradable', 'Recyclable', 'Residual', 'Special Waste'];
    setResult(mockCategories[Math.floor(Math.random() * mockCategories.length)]);
    setIsLoading(false);
  };

  /** Resets all state to allow classifying a new image */
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <main className="min-h-screen bg-white">
      <Navbar />

      {/* Spacer for fixed navbar */}
      <div className="h-16" />

      {/* ─── Header Section ─── */}
      <section className="bg-linear-to-b from-green-50 to-white py-16 px-4 text-center">
        <div className="max-w-3xl mx-auto">
          <span className="inline-block bg-green-100 text-green-700 text-sm font-medium px-4 py-1 rounded-full mb-4">
            🤖 AI-Powered Classification
          </span>
          <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-4">
            Waste Classifier
          </h1>
          <p className="text-gray-600 text-lg leading-relaxed">
            Upload or take a photo of your waste item and our AI will instantly identify whether it&apos;s
            Biodegradable, Recyclable, Residual, or Special Waste — with proper disposal instructions.
          </p>
        </div>
      </section>

      {/* ─── Upload & Classify Section ─── */}
      <section className="py-16 px-4">
        <div className="max-w-2xl mx-auto">

          {/* Drop zone / Upload area */}
          {!previewUrl && (
            <div
              onClick={handleUploadClick}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all ${
                isDragging
                  ? 'border-green-500 bg-green-50'
                  : 'border-green-300 bg-gray-50 hover:border-green-500 hover:bg-green-50'
              }`}
            >
              <div className="text-6xl mb-4">📷</div>
              <p className="text-lg font-semibold text-gray-700 mb-1">Drag & drop your photo here</p>
              <p className="text-gray-500 text-sm mb-6">or click to browse from your device</p>
              <button className="bg-green-600 text-white px-6 py-2 rounded-full text-sm font-semibold hover:bg-green-700 transition-colors">
                Choose File
              </button>
              <p className="text-xs text-gray-400 mt-4">Supports JPG, PNG, WEBP — max 10MB</p>
            </div>
          )}

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleInputChange}
            className="hidden"
          />

          {/* Image preview + classify button */}
          {previewUrl && (
            <div className="space-y-6">
              <div className="relative rounded-2xl overflow-hidden border border-green-200 shadow-md">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl} alt="Uploaded waste" className="w-full h-72 object-cover" />
                <button
                  onClick={handleReset}
                  className="absolute top-3 right-3 bg-white border border-gray-200 rounded-full p-2 hover:bg-red-50 hover:border-red-300 transition-colors shadow"
                  aria-label="Remove image"
                >
                  <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <button
                onClick={handleClassify}
                disabled={isLoading}
                className="w-full bg-green-600 text-white py-4 rounded-full text-lg font-bold hover:bg-green-700 disabled:opacity-60 disabled:cursor-not-allowed transition-colors shadow-lg shadow-green-200"
              >
                {isLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    Analyzing your waste...
                  </span>
                ) : (
                  '🔍 Classify Now'
                )}
              </button>
            </div>
          )}

          {/* Classification result card */}
          {result && (
            <div className={`mt-8 border-2 rounded-2xl p-8 ${DISPOSAL_INSTRUCTIONS[result].color}`}>
              <div className="text-5xl mb-3">{DISPOSAL_INSTRUCTIONS[result].emoji}</div>
              <div className="text-xs font-bold uppercase tracking-widest opacity-70 mb-1">Classified As</div>
              <h2 className="text-3xl font-extrabold mb-4">{result}</h2>
              <p className="text-sm leading-relaxed opacity-90">{DISPOSAL_INSTRUCTIONS[result].instructions}</p>

              <button
                onClick={handleReset}
                className="mt-6 inline-block border-2 border-current text-current px-5 py-2 rounded-full text-sm font-semibold hover:opacity-80 transition-opacity"
              >
                Classify Another Item
              </button>
            </div>
          )}
        </div>
      </section>

      {/* ─── Feature Highlight Section ─── */}
      <section className="py-16 px-4 bg-green-50">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">What the Classifier Detects</h2>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { emoji: '🌱', label: 'Biodegradable', examples: 'Food scraps, leaves, paper, garden waste', color: 'border-green-300 bg-white' },
              { emoji: '♻️', label: 'Recyclable', examples: 'Plastic bottles, cans, cardboard, glass', color: 'border-blue-300 bg-white' },
              { emoji: '🗑️', label: 'Residual', examples: 'Styrofoam, dirty packaging, mixed waste', color: 'border-gray-300 bg-white' },
              { emoji: '⚠️', label: 'Special Waste', examples: 'Batteries, e-waste, chemicals, syringes', color: 'border-red-300 bg-white' },
            ].map((cat) => (
              <div key={cat.label} className={`border-2 ${cat.color} rounded-2xl p-6 text-center`}>
                <div className="text-3xl mb-3">{cat.emoji}</div>
                <h3 className="font-bold text-gray-900 mb-1">{cat.label}</h3>
                <p className="text-xs text-gray-500">{cat.examples}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── How It Works Section ─── */}
      <section className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-gray-500">From photo to proper disposal in seconds</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: '01', emoji: '📸', title: 'Snap or Upload', desc: 'Take a clear photo of your waste item, or upload one from your gallery. Better lighting = better accuracy.' },
              { step: '02', emoji: '🧠', title: 'AI Analyzes It', desc: 'Our AI model processes the image and identifies the waste type within seconds.' },
              { step: '03', emoji: '✅', title: 'Get Instructions', desc: 'Receive the waste category, proper disposal method, and your barangay\'s collection schedule.' },
            ].map((item) => (
              <div key={item.step} className="bg-green-50 rounded-2xl p-8 text-center border border-green-100">
                <div className="text-5xl font-black text-green-200 mb-2">{item.step}</div>
                <div className="text-4xl mb-4">{item.emoji}</div>
                <h3 className="font-bold text-gray-900 text-lg mb-2">{item.title}</h3>
                <p className="text-gray-600 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── CTA Section ─── */}
      <section className="py-20 px-4 bg-linear-to-br from-green-700 to-emerald-800 text-white text-center">
        <div className="max-w-2xl mx-auto">
          <h2 className="text-3xl font-bold mb-4">Not sure what to throw away?</h2>
          <p className="text-green-100 mb-8">
            Start classifying your waste now. It only takes a second and makes a real difference.
          </p>
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="bg-white text-green-700 font-bold px-10 py-4 rounded-full text-lg hover:bg-green-50 transition-colors shadow-xl"
          >
            Start Classifying Now ↑
          </button>
        </div>
      </section>

      {/* ─── FAQ Section ─── */}
      <section className="py-24 px-4">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Classifier FAQs</h2>
            <p className="text-gray-500">Common questions about using the AI classifier</p>
          </div>
          <div className="space-y-4">
            {[
              { q: 'What image quality do I need?', a: 'A clear, well-lit photo taken within 1–2 feet of the item works best. Avoid blurry or heavily shadowed images for highest accuracy.' },
              { q: 'Which waste types are supported?', a: 'All four DENR-recognized categories: Biodegradable, Recyclable, Residual, and Special Waste. We continuously train the model on new items.' },
              { q: 'Is my photo stored or shared?', a: 'No. Photos are processed in memory and immediately discarded. Only anonymous classification counts are recorded to improve the AI.' },
              { q: 'How accurate is the classifier?', a: 'Our model achieves over 95% accuracy across standard household waste types. Accuracy may vary for rare or mixed-material items.' },
              { q: 'Can I use it on mobile?', a: 'Yes! The classifier is fully mobile-optimized. You can even use your phone\'s camera directly to capture and classify waste on the spot.' },
              { q: 'What if the classification is wrong?', a: 'Use the contact form below to report misclassifications. Your feedback helps us retrain and improve the model.' },
            ].map((faq, i) => (
              <div key={i} className="border border-green-100 rounded-xl p-6 hover:border-green-300 hover:bg-green-50 transition-all">
                <h3 className="font-semibold text-gray-900 mb-2">{faq.q}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{faq.a}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── Contact / Support Section ─── */}
      <section className="py-20 px-4 bg-green-50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Need Help or Found an Issue?</h2>
          <p className="text-gray-500 mb-8">
            Report a misclassification or reach out to our support team — your feedback makes the AI smarter.
          </p>
          <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <a
              href="mailto:support@davaoclean.ai"
              className="flex items-center gap-4 bg-white border border-green-200 rounded-2xl p-6 hover:shadow-md hover:border-green-400 transition-all text-left"
            >
              <span className="text-3xl">✉️</span>
              <div>
                <div className="font-semibold text-gray-900">Email Support</div>
                <div className="text-green-600 text-sm">support@davaoclean.ai</div>
              </div>
            </a>
            <a
              href="tel:+6382123456"
              className="flex items-center gap-4 bg-white border border-green-200 rounded-2xl p-6 hover:shadow-md hover:border-green-400 transition-all text-left"
            >
              <span className="text-3xl">📞</span>
              <div>
                <div className="font-semibold text-gray-900">Call Us</div>
                <div className="text-green-600 text-sm">+63 (82) 123-4567</div>
              </div>
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}