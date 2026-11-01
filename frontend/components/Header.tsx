'use client';

import React, { useEffect, useState } from 'react';
import { checkBackendHealth, HealthResponse, API_BASE_URL } from '@/lib/api';
import { Mic2, Radio, Sparkles, Cpu } from 'lucide-react';

export default function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      try {
        const h = await checkBackendHealth();
        if (isMounted) {
          setHealth(h);
          setOnline(true);
        }
      } catch (e) {
        if (isMounted) {
          setOnline(false);
        }
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Mic2 className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white">VocalMorph</h1>
              <span className="text-[10px] font-semibold tracking-wider px-2.5 py-0.5 rounded-full bg-gradient-to-r from-indigo-500/10 to-violet-500/10 text-indigo-300 border border-indigo-500/20 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-indigo-400" />
                Hybrid (Zero-Shot + RVC)
              </span>
            </div>
            <p className="text-xs text-slate-400">Singing Voice Conversion & Demucs Separation</p>
          </div>
        </div>

        {/* Backend Status indicator */}
        <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700">
          <Radio className={`w-3.5 h-3.5 ${online ? 'text-emerald-400 animate-pulse' : 'text-rose-400'}`} />
          <span className="text-slate-300">
            {online === null ? (
              'Checking backend...'
            ) : online ? (
              <span>
                Colab GPU:{' '}
                <strong className="text-emerald-400 font-semibold">{health?.gpu_name || 'Active'}</strong>
              </span>
            ) : (
              <span className="text-rose-400 font-medium">Backend Offline</span>
            )}
          </span>
        </div>
      </div>
    </header>
  );
}
