'use client';

import React, { useRef, useState } from 'react';
import { Music, UploadCloud, X, Play, Pause } from 'lucide-react';

interface SongUploaderProps {
  onSongSelected: (file: File | null) => void;
  selectedSong: File | null;
}

export default function SongUploader({ onSongSelected, selectedSong }: SongUploaderProps) {
  const [duration, setDuration] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFile = (file: File) => {
    setError(null);
    if (!file.type.startsWith('audio/') && !file.name.match(/\.(mp3|wav|flac|m4a|ogg)$/i)) {
      setError('Please select a valid audio file (.mp3, .wav, .flac, .m4a)');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size exceeds the 50MB limit.');
      return;
    }

    // Inspect duration
    const url = URL.createObjectURL(file);
    const tempAudio = new Audio(url);
    tempAudio.onloadedmetadata = () => {
      if (tempAudio.duration > 360) {
        setError('Song duration exceeds 6 minutes. Please choose a shorter track.');
        onSongSelected(null);
        setDuration(null);
      } else {
        setDuration(tempAudio.duration);
        onSongSelected(file);
      }
    };
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const togglePreview = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl relative backdrop-blur-sm">
      <div className="flex items-center gap-3 mb-4">
        <div className="h-8 w-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/20">
          1
        </div>
        <h2 className="text-lg font-semibold text-white">Select Song to Convert</h2>
      </div>

      {!selectedSong ? (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 hover:border-indigo-500/60 transition-colors rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-slate-950/40 text-center group"
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="audio/*,.mp3,.wav,.flac,.m4a"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="h-12 w-12 rounded-full bg-slate-800 group-hover:bg-indigo-600/20 transition-colors flex items-center justify-center mb-3 text-slate-400 group-hover:text-indigo-400">
            <UploadCloud className="w-6 h-6" />
          </div>
          <p className="text-sm font-medium text-slate-200">
            Click to upload or drag & drop audio track
          </p>
          <p className="text-xs text-slate-400 mt-1">MP3, WAV, FLAC, M4A (Max 6 min, 50MB)</p>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={togglePreview}
              className="h-10 w-10 rounded-full bg-indigo-600 hover:bg-indigo-500 transition-colors flex items-center justify-center text-white shrink-0 shadow-md shadow-indigo-600/30"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div className="min-w-0">
              <p className="text-sm font-medium text-white truncate">{selectedSong.name}</p>
              <p className="text-xs text-slate-400">
                {(selectedSong.size / (1024 * 1024)).toFixed(1)} MB
                {duration ? ` • ${formatTime(duration)}` : ''}
              </p>
            </div>
            <audio
              ref={audioRef}
              src={selectedSong ? URL.createObjectURL(selectedSong) : undefined}
              onEnded={() => setIsPlaying(false)}
            />
          </div>

          <button
            onClick={() => {
              if (audioRef.current) audioRef.current.pause();
              setIsPlaying(false);
              onSongSelected(null);
              setDuration(null);
            }}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {error && (
        <div className="mt-3 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          {error}
        </div>
      )}
    </div>
  );
}
