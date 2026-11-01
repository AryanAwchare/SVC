'use client';

import React, { useState, useEffect } from 'react';
import Header from '@/components/Header';
import SongUploader from '@/components/SongUploader';
import VoiceRecorder from '@/components/VoiceRecorder';
import SettingsPanel from '@/components/SettingsPanel';
import ProgressCard from '@/components/ProgressCard';
import ABPlayer from '@/components/ABPlayer';
import DownloadCard from '@/components/DownloadCard';
import {
  createConversionJob,
  uploadAudioFile,
  startConversion,
  pollJobStatus,
  getAvailableModels,
  JobOptions,
  JobStatusResponse,
  VoiceModelInfo,
} from '@/lib/api';

export default function Home() {
  const [engine, setEngine] = useState<'seed-vc' | 'rvc'>('seed-vc');
  const [selectedModelId, setSelectedModelId] = useState<string>('zero-shot');
  const [availableModels, setAvailableModels] = useState<VoiceModelInfo[]>([]);

  const [selectedSong, setSelectedSong] = useState<File | null>(null);
  const [voiceSample, setVoiceSample] = useState<Blob | File | null>(null);
  const [consent, setConsent] = useState<boolean>(false);
  const [options, setOptions] = useState<JobOptions>({
    semitone_shift: 0,
    diffusion_steps: 55,
    output_format: 'mp3',
    rvc_index_rate: 0.75,
  });

  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAvailableModels().then((models) => {
      setAvailableModels(models);
    });
  }, []);

  const requiresVoiceSample = engine === 'seed-vc' || selectedModelId === 'custom_voice';
  const canSubmit =
    !!selectedSong &&
    consent &&
    !isProcessing &&
    (!requiresVoiceSample || !!voiceSample);

  const handleSubmit = async () => {
    if (!selectedSong || !consent) return;
    if (requiresVoiceSample && !voiceSample) return;

    setError(null);
    setIsProcessing(true);
    setJobStatus(null);

    try {
      // 1. Create Job record specifying hybrid engine & model
      const jobRes = await createConversionJob(options, engine, selectedModelId);
      const jId = jobRes.job_id;
      setActiveJobId(jId);

      // 2. Upload Song File
      await uploadAudioFile(jId, 'song', selectedSong, selectedSong.name);

      // 3. Upload Voice Sample if required
      if (voiceSample) {
        const sampleFilename = voiceSample instanceof File ? voiceSample.name : 'recorded_sample.wav';
        await uploadAudioFile(jId, 'sample', voiceSample, sampleFilename);
      }

      // 4. Start Pipeline on GPU
      await startConversion(jId, consent);

      // 5. Poll status until finished
      const pollInterval = setInterval(async () => {
        try {
          const status = await pollJobStatus(jId);
          setJobStatus(status);

          if (status.status === 'DONE') {
            clearInterval(pollInterval);
            setIsProcessing(false);
          } else if (status.status === 'FAILED') {
            clearInterval(pollInterval);
            setIsProcessing(false);
            setError(status.error || status.message || 'Conversion failed on GPU.');
          }
        } catch (e: any) {
          console.error('Polling error:', e);
        }
      }, 2500);
    } catch (err: any) {
      console.error('Job submission failed:', err);
      setIsProcessing(false);
      setError(err.message || 'Could not connect to Colab GPU backend.');
    }
  };

  const resetAll = () => {
    setSelectedSong(null);
    setVoiceSample(null);
    setConsent(false);
    setActiveJobId(null);
    setJobStatus(null);
    setIsProcessing(false);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col justify-between">
      <div>
        <Header />

        <main className="max-w-4xl mx-auto px-4 py-8 sm:py-12">
          {/* Hero Banner */}
          {!isProcessing && jobStatus?.status !== 'DONE' && (
            <div className="text-center mb-10">
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white mb-3">
                Transform Any Song with{' '}
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-indigo-200 to-violet-400">
                  Hybrid Voice AI
                </span>
              </h2>
              <p className="text-sm sm:text-base text-slate-400 max-w-xl mx-auto">
                {engine === 'seed-vc'
                  ? 'Option A: Instant Zero-Shot Conversion. Record 15s to hear any track in your voice with zero waiting.'
                  : 'Option B: Studio RVC Mode. Powered by trained neural models for 95%+ studio-grade vocal likeness.'}
              </p>
            </div>
          )}

          {/* Workflow View 1: Configuration & Upload */}
          {!isProcessing && jobStatus?.status !== 'DONE' && (
            <div className="space-y-6">
              {/* Step 1: Song File */}
              <SongUploader
                onSongSelected={setSelectedSong}
                selectedSong={selectedSong}
              />

              {/* Step 2: Voice Reference (Adaptive depending on Engine & Model) */}
              {requiresVoiceSample ? (
                <VoiceRecorder
                  onSampleReady={setVoiceSample}
                  sampleReady={!!voiceSample}
                />
              ) : (
                <div className="p-5 rounded-2xl bg-violet-950/20 border border-violet-800/30 text-xs text-violet-300 flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-white">Using Pre-Trained Studio Voice</p>
                    <p className="text-violet-400 mt-0.5">
                      No recording required. This model has already been fine-tuned on hundreds of vocal takes.
                    </p>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-violet-500/20 text-violet-200 font-medium">
                    Ready
                  </span>
                </div>
              )}

              {/* Step 3: Engine Selection & Settings Panel */}
              <SettingsPanel
                engine={engine}
                onEngineChange={(eng) => {
                  setEngine(eng);
                  if (eng === 'rvc') {
                    setSelectedModelId('custom_voice');
                  } else {
                    setSelectedModelId('zero-shot');
                  }
                }}
                selectedModelId={selectedModelId}
                onModelIdChange={setSelectedModelId}
                availableModels={availableModels}
                options={options}
                onOptionsChange={setOptions}
                consent={consent}
                onConsentChange={setConsent}
                canSubmit={canSubmit}
                onSubmit={handleSubmit}
              />

              {error && (
                <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs text-center">
                  {error}
                </div>
              )}
            </div>
          )}

          {/* Workflow View 2: Live Processing */}
          {isProcessing && (
            <ProgressCard
              status={jobStatus}
              error={error}
              onRetry={resetAll}
            />
          )}

          {/* Workflow View 3: Completed Results & Comparison Player */}
          {!isProcessing && jobStatus?.status === 'DONE' && jobStatus.outputs && (
            <div>
              <div className="text-center mb-8">
                <span className="text-xs font-semibold px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Conversion Complete • {engine === 'seed-vc' ? 'Seed-VC 44.1kHz DiT' : 'Studio RVC v2'}
                </span>
                <h3 className="text-2xl font-bold text-white mt-2">Listen & Compare</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Seamlessly toggle between the original lead vocal and your AI-rendered mix.
                </p>
              </div>

              <ABPlayer
                originalUrl={jobStatus.outputs.original_song_url}
                convertedUrl={jobStatus.outputs.final_mix_url}
              />

              <DownloadCard
                jobId={jobStatus.job_id}
                finalMixUrl={jobStatus.outputs.final_mix_url}
                vocalsUrl={jobStatus.outputs.vocals_url}
                onReset={resetAll}
              />
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <p>
          VocalMorph • Hybrid Engine (Seed-VC Zero-Shot + Studio RVC v2) & Demucs FT • For research & creative use.
        </p>
      </footer>
    </div>
  );
}
