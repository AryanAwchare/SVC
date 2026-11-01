'use client';

import React from 'react';
import { Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { JobStatusResponse } from '@/lib/api';

interface ProgressCardProps {
  status: JobStatusResponse | null;
  error: string | null;
  onRetry: () => void;
}

export default function ProgressCard({ status, error, onRetry }: ProgressCardProps) {
  const stages = [
    { key: 'VALIDATING', label: 'Validating Audio', minProgress: 5 },
    { key: 'SEPARATING', label: 'Separating Lead Vocals', minProgress: 20 },
    { key: 'CONVERTING', label: status?.engine === 'rvc' ? 'Studio RVC Neural Conversion' : 'Seed-VC Voice Conversion', minProgress: 45 },
    { key: 'MIXING', label: 'Remixing & EBU-R128 Mastering', minProgress: 85 },
    { key: 'DONE', label: 'Completed', minProgress: 100 },
  ];

  const currentProgress = status?.progress || 5;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 shadow-2xl backdrop-blur-md max-w-xl mx-auto my-8">
      <div className="text-center mb-6">
        <div className="inline-flex items-center justify-center h-14 w-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-4 shadow-lg shadow-indigo-500/10">
          {!error ? (
            <Loader2 className="w-7 h-7 animate-spin text-indigo-400" />
          ) : (
            <AlertCircle className="w-7 h-7 text-rose-400" />
          )}
        </div>
        <h3 className="text-xl font-bold text-white">
          {!error ? 'Converting Your Song...' : 'Conversion Interrupted'}
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          {!error
            ? status?.message || 'Processing audio on Google Colab GPU'
            : error}
        </p>
      </div>

      {!error ? (
        <>
          {/* Progress Bar */}
          <div className="w-full bg-slate-950 rounded-full h-3 p-0.5 border border-slate-800 mb-6 overflow-hidden">
            <div
              className="bg-gradient-to-r from-indigo-500 to-violet-500 h-full rounded-full transition-all duration-500 ease-out shadow-sm shadow-indigo-500/50"
              style={{ width: `${Math.max(5, currentProgress)}%` }}
            />
          </div>

          {/* Stepper */}
          <div className="space-y-3">
            {stages.map((st, idx) => {
              const isPast = currentProgress > st.minProgress;
              const isCurrent =
                currentProgress >= st.minProgress &&
                (idx === stages.length - 1 || currentProgress < stages[idx + 1].minProgress);

              return (
                <div key={st.key} className="flex items-center gap-3">
                  <div className="shrink-0">
                    {isPast ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    ) : isCurrent ? (
                      <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border border-slate-700 bg-slate-800/50" />
                    )}
                  </div>
                  <span
                    className={`text-xs font-medium transition-colors ${
                      isCurrent
                        ? 'text-white font-semibold'
                        : isPast
                        ? 'text-slate-300'
                        : 'text-slate-600'
                    }`}
                  >
                    {st.label}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-8 pt-4 border-t border-slate-800/80 flex justify-between text-xs text-slate-500">
            <span>Elapsed: {status?.elapsed_seconds || 0}s</span>
            <span>Running on Colab GPU</span>
          </div>
        </>
      ) : (
        <div className="text-center pt-2">
          <p className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-xl p-4 mb-4">
            {error}
          </p>
          <button
            onClick={onRetry}
            className="px-6 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold transition-colors"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}
