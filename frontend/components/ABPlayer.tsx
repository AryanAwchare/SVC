'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Sparkles, Music2, Loader2, AlertCircle } from 'lucide-react';
import { API_BASE_URL } from '@/lib/api';

interface ABPlayerProps {
  originalUrl: string;
  convertedUrl: string;
}

export default function ABPlayer({ originalUrl, convertedUrl }: ABPlayerProps) {
  const [activeTrack, setActiveTrack] = useState<'original' | 'converted'>('converted');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.85);
  const [isMuted, setIsMuted] = useState(false);

  // Blob URLs to bypass ngrok browser warning
  const [originalBlobUrl, setOriginalBlobUrl] = useState<string | null>(null);
  const [convertedBlobUrl, setConvertedBlobUrl] = useState<string | null>(null);
  const [isLoadingBlobs, setIsLoadingBlobs] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Fetch audio files as Blobs using the ngrok-skip-browser-warning header
  useEffect(() => {
    let isMounted = true;
    let origBlob: string | null = null;
    let convBlob: string | null = null;

    async function loadAudioBlobs() {
      setIsLoadingBlobs(true);
      setLoadError(null);

      try {
        const fullOrig = originalUrl.startsWith('http') ? originalUrl : `${API_BASE_URL}${originalUrl}`;
        const fullConv = convertedUrl.startsWith('http') ? convertedUrl : `${API_BASE_URL}${convertedUrl}`;

        const [resOrig, resConv] = await Promise.all([
          fetch(fullOrig, {
            headers: { 'ngrok-skip-browser-warning': 'true' },
          }),
          fetch(fullConv, {
            headers: { 'ngrok-skip-browser-warning': 'true' },
          }),
        ]);

        if (!resOrig.ok || !resConv.ok) {
          throw new Error('Failed to download audio stems from server.');
        }

        const [blobOrig, blobConv] = await Promise.all([
          resOrig.blob(),
          resConv.blob(),
        ]);

        origBlob = URL.createObjectURL(blobOrig);
        convBlob = URL.createObjectURL(blobConv);

        if (isMounted) {
          setOriginalBlobUrl(origBlob);
          setConvertedBlobUrl(convBlob);
          setIsLoadingBlobs(false);
        }
      } catch (err: any) {
        console.error('Audio loading error:', err);
        if (isMounted) {
          setLoadError('Could not load audio streams. Please check your network connection.');
          setIsLoadingBlobs(false);
        }
      }
    }

    loadAudioBlobs();

    return () => {
      isMounted = false;
      if (origBlob) URL.revokeObjectURL(origBlob);
      if (convBlob) URL.revokeObjectURL(convBlob);
    };
  }, [originalUrl, convertedUrl]);

  // Switch between tracks while preserving exact playback position
  const switchTrack = (target: 'original' | 'converted') => {
    if (target === activeTrack) return;
    const currentPos = audioRef.current?.currentTime || 0;
    const wasPlaying = isPlaying;

    setActiveTrack(target);

    setTimeout(() => {
      if (audioRef.current) {
        audioRef.current.currentTime = currentPos;
        if (wasPlaying) {
          audioRef.current.play().catch(() => {});
        }
      }
    }, 50);
  };

  const togglePlay = () => {
    if (!audioRef.current || isLoadingBlobs) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch((err) => {
        console.warn('Playback blocked or failed:', err);
      });
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs) || secs < 0) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  const currentAudioSource = activeTrack === 'converted' ? convertedBlobUrl : originalBlobUrl;

  return (
    <div className="bg-gradient-to-b from-slate-900/90 to-slate-950/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-md">
      {currentAudioSource && (
        <audio
          ref={audioRef}
          src={currentAudioSource}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />
      )}

      {/* A / B Selector Switch */}
      <div className="flex justify-center mb-8">
        <div className="bg-slate-950 p-1.5 rounded-2xl border border-slate-800 flex gap-2 shadow-inner">
          <button
            onClick={() => switchTrack('original')}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-xs transition-all ${
              activeTrack === 'original'
                ? 'bg-slate-800 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Music2 className="w-4 h-4" />
            <span>Original Lead Vocal</span>
          </button>

          <button
            onClick={() => switchTrack('converted')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-xs transition-all ${
              activeTrack === 'converted'
                ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-4 h-4 text-indigo-200" />
            <span>Your Voice Remix</span>
          </button>
        </div>
      </div>

      {/* Loading Blobs State */}
      {isLoadingBlobs && (
        <div className="flex items-center justify-center gap-3 py-4 mb-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium">
          <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
          <span>Buffering high-fidelity audio streams from GPU...</span>
        </div>
      )}

      {/* Error State */}
      {loadError && (
        <div className="flex items-center justify-center gap-2 py-3 mb-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 text-rose-400" />
          <span>{loadError}</span>
        </div>
      )}

      {/* Scrubber Timeline */}
      <div className="mb-6">
        <input
          type="range"
          min="0"
          max={duration || 100}
          step="0.1"
          value={currentTime}
          onChange={handleSeek}
          disabled={isLoadingBlobs || !!loadError}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs font-mono text-slate-400 mt-2">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="flex items-center justify-between">
        {/* Play/Pause */}
        <div className="flex items-center gap-4">
          <button
            onClick={togglePlay}
            disabled={isLoadingBlobs || !!loadError}
            className="h-14 w-14 rounded-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:cursor-not-allowed transition-transform active:scale-95 shadow-xl shadow-indigo-600/30 flex items-center justify-center text-white"
          >
            {isLoadingBlobs ? (
              <Loader2 className="w-6 h-6 animate-spin text-indigo-300" />
            ) : isPlaying ? (
              <Pause className="w-6 h-6" />
            ) : (
              <Play className="w-6 h-6 ml-0.5" />
            )}
          </button>
          <div>
            <p className="text-sm font-semibold text-white">
              {activeTrack === 'converted' ? 'Your Voice Converted Mix' : 'Original Artist Song'}
            </p>
            <p className="text-xs text-slate-400">
              {activeTrack === 'converted'
                ? 'Seed-VC 44.1kHz • Normalized to -14 LUFS'
                : 'Original vocal stem & mix'}
            </p>
          </div>
        </div>

        {/* Volume */}
        <div className="hidden sm:flex items-center gap-2">
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="text-slate-400 hover:text-white p-2"
          >
            {isMuted || volume === 0 ? (
              <VolumeX className="w-4 h-4" />
            ) : (
              <Volume2 className="w-4 h-4" />
            )}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={isMuted ? 0 : volume}
            onChange={(e) => {
              setVolume(parseFloat(e.target.value));
              setIsMuted(false);
            }}
            className="w-20 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
          />
        </div>
      </div>
    </div>
  );
}
