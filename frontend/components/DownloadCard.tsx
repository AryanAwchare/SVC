'use client';

import React, { useState } from 'react';
import { Download, Trash2, RotateCcw, Check, Sparkles, Loader2 } from 'lucide-react';
import { API_BASE_URL, deleteJobData } from '@/lib/api';

interface DownloadCardProps {
  jobId: string;
  finalMixUrl: string;
  vocalsUrl: string;
  onReset: () => void;
}

export default function DownloadCard({
  jobId,
  finalMixUrl,
  vocalsUrl,
  onReset,
}: DownloadCardProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleted, setDeleted] = useState(false);
  const [downloading, setDownloading] = useState<'mix' | 'vocals' | null>(null);

  const handleDownload = async (url: string, filename: string, type: 'mix' | 'vocals') => {
    const fullUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;
    setDownloading(type);

    try {
      const res = await fetch(fullUrl, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
      });

      if (!res.ok) throw new Error('Download failed');

      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error('Download error:', err);
      alert('Could not download audio file. Please try again.');
    } finally {
      setDownloading(null);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to permanently delete your audio files from the server?')) {
      return;
    }
    setIsDeleting(true);
    try {
      await deleteJobData(jobId);
      setDeleted(true);
      setTimeout(() => {
        onReset();
      }, 1500);
    } catch (e) {
      alert('Could not delete files from server.');
      setIsDeleting(false);
    }
  };

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm mt-6">
      <h4 className="text-sm font-semibold text-white mb-4">Export & Downloads</h4>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
        {/* Full Mix Download */}
        <button
          onClick={() =>
            handleDownload(
              finalMixUrl,
              `vocalmorph_${jobId.slice(0, 8)}_full_mix.mp3`,
              'mix'
            )
          }
          disabled={downloading !== null}
          className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-xs transition-colors shadow-lg shadow-indigo-600/20"
        >
          {downloading === 'mix' ? (
            <Loader2 className="w-4 h-4 animate-spin text-white" />
          ) : (
            <Download className="w-4 h-4" />
          )}
          <span>{downloading === 'mix' ? 'Saving MP3...' : 'Download Full Song (MP3)'}</span>
        </button>

        {/* Vocals Stem Download */}
        <button
          onClick={() =>
            handleDownload(
              vocalsUrl,
              `vocalmorph_${jobId.slice(0, 8)}_vocals.wav`,
              'vocals'
            )
          }
          disabled={downloading !== null}
          className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 font-semibold text-xs border border-slate-700 transition-colors"
        >
          {downloading === 'vocals' ? (
            <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
          ) : (
            <Sparkles className="w-4 h-4 text-indigo-400" />
          )}
          <span>{downloading === 'vocals' ? 'Saving WAV...' : 'Download Vocal Stem (WAV)'}</span>
        </button>
      </div>

      {/* Expiration Notice & Privacy Wipe */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800 text-xs text-slate-400">
        <p className="text-center sm:text-left">
          Files are automatically deleted from Colab storage after 24 hours.
        </p>

        <div className="flex items-center gap-3">
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Convert Another</span>
          </button>

          <button
            onClick={handleDelete}
            disabled={isDeleting || deleted}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 font-medium transition-colors"
          >
            {deleted ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Purged!</span>
              </>
            ) : (
              <>
                <Trash2 className="w-3.5 h-3.5" />
                <span>{isDeleting ? 'Deleting...' : 'Delete My Audio Now'}</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
