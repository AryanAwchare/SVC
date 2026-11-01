'use client';

import React, { useState } from 'react';
import { Settings2, Zap, Award, ChevronDown, ChevronUp, Sliders, Music, Sparkles } from 'lucide-react';
import { JobOptions, VoiceModelInfo } from '@/lib/api';

interface SettingsPanelProps {
  engine: 'seed-vc' | 'rvc';
  onEngineChange: (engine: 'seed-vc' | 'rvc') => void;
  selectedModelId: string;
  onModelIdChange: (modelId: string) => void;
  availableModels: VoiceModelInfo[];
  options: JobOptions;
  onOptionsChange: (opts: JobOptions) => void;
  consent: boolean;
  onConsentChange: (c: boolean) => void;
  canSubmit: boolean;
  onSubmit: () => void;
}

export default function SettingsPanel({
  engine,
  onEngineChange,
  selectedModelId,
  onModelIdChange,
  availableModels,
  options,
  onOptionsChange,
  consent,
  onConsentChange,
  canSubmit,
  onSubmit,
}: SettingsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl relative backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/20">
            3
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">AI Engine & Voice Settings</h2>
            <p className="text-xs text-slate-400">Choose between Instant Zero-Shot or Studio Trained Mode</p>
          </div>
        </div>

        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
        >
          <span>{isOpen ? 'Collapse Options' : 'Customize Audio'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {/* Engine Switcher (Option A vs Option B) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
        {/* Option A: Seed-VC Zero-Shot */}
        <button
          type="button"
          onClick={() => onEngineChange('seed-vc')}
          className={`p-4 rounded-xl border text-left transition-all relative ${
            engine === 'seed-vc'
              ? 'border-indigo-500 bg-gradient-to-br from-indigo-950/60 to-slate-900/80 shadow-md ring-1 ring-indigo-500/50'
              : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="flex items-center gap-2 font-bold text-sm text-white">
              <Zap className="w-4 h-4 text-amber-400" />
              Instant Morph (Seed-VC)
            </span>
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Zero-Shot
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Record a 15–20s voice clip. Converts in ~45s with <strong>0 waiting or training required</strong>.
          </p>
        </button>

        {/* Option B: RVC v2 Studio Trained */}
        <button
          type="button"
          onClick={() => onEngineChange('rvc')}
          className={`p-4 rounded-xl border text-left transition-all relative ${
            engine === 'rvc'
              ? 'border-violet-500 bg-gradient-to-br from-violet-950/60 to-slate-900/80 shadow-md ring-1 ring-violet-500/50'
              : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="flex items-center gap-2 font-bold text-sm text-white">
              <Award className="w-4 h-4 text-violet-400" />
              Studio Clone (RVC v2)
            </span>
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-violet-500/10 text-violet-300 border border-violet-500/20">
              95%+ Likeness
            </span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Fine-tuned neural model with uncanny timbre mimicry. Choose an AI artist or your custom model.
          </p>
        </button>
      </div>

      {/* RVC Model Selector (Shown when RVC is active) */}
      {engine === 'rvc' && (
        <div className="mb-6 p-4 rounded-xl bg-violet-950/20 border border-violet-800/30">
          <label className="block text-xs font-semibold text-violet-200 mb-2">
            Select Trained Voice Model
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {[
              { id: 'custom_voice', name: 'My Trained Voice Model', desc: 'Loaded from Colab models/rvc/' },
              { id: 'bruno_mars', name: 'Bruno Mars (AI Studio)', desc: 'Pre-trained RVC v2 checkpoint' },
              { id: 'taylor_swift', name: 'Taylor Swift (AI Studio)', desc: 'Pre-trained RVC v2 checkpoint' },
              { id: 'drake', name: 'Drake (AI Studio)', desc: 'Pre-trained RVC v2 checkpoint' },
            ].map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => onModelIdChange(m.id)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  selectedModelId === m.id
                    ? 'border-violet-500 bg-violet-600/20 text-white'
                    : 'border-slate-800/80 bg-slate-950/60 text-slate-400 hover:border-slate-700'
                }`}
              >
                <p className="font-semibold text-xs text-slate-200">{m.name}</p>
                <p className="text-[10px] text-slate-400 mt-0.5">{m.desc}</p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Accordion Content for Deep Audio Tuning */}
      {isOpen && (
        <div className="space-y-6 pt-2 pb-4 border-b border-slate-800 text-sm">
          {/* Key / Pitch Shift */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-medium text-slate-300">
                Key / Pitch Transposition (Semitones)
              </label>
              <span className="text-xs font-mono font-bold text-indigo-400 px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20">
                {options.semitone_shift > 0 ? `+${options.semitone_shift}` : options.semitone_shift} semitones
              </span>
            </div>
            <input
              type="range"
              min="-12"
              max="12"
              step="1"
              value={options.semitone_shift}
              onChange={(e) =>
                onOptionsChange({ ...options, semitone_shift: parseInt(e.target.value) })
              }
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1">
              <span>-12 (Female ➔ Male Octave)</span>
              <span>0 (Auto-Detect / Same Octave)</span>
              <span>+12 (Male ➔ Female Octave)</span>
            </div>
          </div>

          {/* Seed-VC Specific Options */}
          {engine === 'seed-vc' && (
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-2">
                Diffusion Steps (Quality vs. Speed)
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: 'Standard', steps: 35, desc: 'Fast render (~35s)' },
                  { label: 'High Fidelity', steps: 55, desc: 'Balanced detail (Default)' },
                  { label: 'Studio Master', steps: 75, desc: 'Maximum acoustic precision' },
                ].map((q) => (
                  <button
                    key={q.label}
                    type="button"
                    onClick={() => onOptionsChange({ ...options, diffusion_steps: q.steps })}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      options.diffusion_steps === q.steps
                        ? 'border-indigo-500 bg-indigo-600/15 text-white shadow-sm'
                        : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <p className="font-semibold text-xs text-white">{q.label}</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">{q.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* RVC Specific Options */}
          {engine === 'rvc' && (
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-xs font-medium text-slate-300">
                  Feature Retrieval Ratio (Index Rate)
                </label>
                <span className="text-xs font-mono font-bold text-violet-400 px-2 py-0.5 rounded bg-violet-500/10 border border-violet-500/20">
                  {(options.rvc_index_rate ?? 0.75).toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={options.rvc_index_rate ?? 0.75}
                onChange={(e) =>
                  onOptionsChange({ ...options, rvc_index_rate: parseFloat(e.target.value) })
                }
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
              />
              <p className="text-[10px] text-slate-500 mt-1">
                Higher rates prevent voice cracking and force stronger voice similarity to the trained dataset.
              </p>
            </div>
          )}

          {/* Output Format */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-2">Output Format</label>
            <div className="flex gap-3">
              {(['mp3', 'wav'] as const).map((fmt) => (
                <button
                  key={fmt}
                  type="button"
                  onClick={() => onOptionsChange({ ...options, output_format: fmt })}
                  className={`px-4 py-1.5 rounded-lg border text-xs font-medium uppercase tracking-wider transition-all ${
                    options.output_format === fmt
                      ? 'border-indigo-500 bg-indigo-600 text-white shadow-sm'
                      : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {fmt}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Mandatory Consent Checkbox */}
      <div className="mt-4 pt-2">
        <label className="flex items-start gap-3 cursor-pointer group">
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => onConsentChange(e.target.checked)}
            className="mt-0.5 h-4 w-4 rounded border-slate-700 bg-slate-950 text-indigo-600 focus:ring-indigo-500/30"
          />
          <div className="text-xs text-slate-400 group-hover:text-slate-300 transition-colors">
            <span className="font-medium text-slate-200">Legal & Ethics Consent: </span>
            I certify that this voice sample is my own, or I have explicit authorization from the voice owner. I have the legal right to process this audio.
          </div>
        </label>
      </div>

      {/* Submit Button */}
      <div className="mt-6">
        <button
          onClick={onSubmit}
          disabled={!canSubmit}
          className={`w-full py-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-xl ${
            canSubmit
              ? 'bg-gradient-to-r from-indigo-600 via-indigo-500 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-indigo-600/25 cursor-pointer scale-[1.01]'
              : 'bg-slate-800 text-slate-500 cursor-not-allowed'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          <span>
            {engine === 'seed-vc' ? 'Morph Singing Voice (Instant Zero-Shot)' : 'Render Song with Studio RVC Model'}
          </span>
        </button>
      </div>
    </div>
  );
}
