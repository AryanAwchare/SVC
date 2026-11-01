'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, RotateCcw, Upload, CheckCircle2, Play, Pause } from 'lucide-react';

interface VoiceRecorderProps {
  onSampleReady: (blob: Blob | File | null) => void;
  sampleReady: boolean;
}

export default function VoiceRecorder({ onSampleReady, sampleReady }: VoiceRecorderProps) {
  const [mode, setMode] = useState<'record' | 'upload'>('record');
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const audioPreviewRef = useRef<HTMLAudioElement | null>(null);

  // Stop recording and cleanup
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
    if (timerRef.current) clearInterval(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      setRecordDuration(0);

      // Volume meter setup
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 64;
      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateVolume = () => {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        setVolumeLevel(Math.min(100, Math.round((avg / 128) * 100)));
        animFrameRef.current = requestAnimationFrame(updateVolume);
      };
      updateVolume();

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        onSampleReady(audioBlob);

        // Stop stream tracks
        stream.getTracks().forEach((track) => track.stop());
        if (audioCtx.state !== 'closed') audioCtx.close();
      };

      mediaRecorder.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      alert('Microphone access denied or not available. Please allow mic permissions.');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      onSampleReady(file);
    }
  };

  const resetSample = () => {
    setAudioUrl(null);
    onSampleReady(null);
    setRecordDuration(0);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (!audioPreviewRef.current) return;
    if (isPlaying) {
      audioPreviewRef.current.pause();
      setIsPlaying(false);
    } else {
      audioPreviewRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-xl relative backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/20">
            2
          </div>
          <h2 className="text-lg font-semibold text-white">Your Voice Sample</h2>
        </div>

        {/* Tab switcher */}
        <div className="flex bg-slate-950/60 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setMode('record')}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              mode === 'record' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Record Mic
          </button>
          <button
            onClick={() => setMode('upload')}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              mode === 'upload' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Upload Clip
          </button>
        </div>
      </div>

      {!audioUrl ? (
        mode === 'record' ? (
          <div className="flex flex-col items-center justify-center py-6 text-center">
            {/* Reading Prompt Box */}
            <div className="w-full bg-slate-950/50 border border-slate-800 rounded-xl p-4 mb-5 text-left">
              <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400">
                Suggested 15s Reading Prompt:
              </span>
              <p className="text-xs text-slate-300 mt-1 italic leading-relaxed">
                "The summer sun reflected gently on the calm water, while children laughed along the shore and the warm breeze carried distant melodies."
              </p>
            </div>

            {/* Live Volume Meter when recording */}
            {isRecording && (
              <div className="w-full max-w-xs mb-4">
                <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                  <span>Input Volume</span>
                  <span>{recordDuration}s / 30s recommended</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-indigo-500 transition-all duration-75"
                    style={{ width: `${Math.min(100, volumeLevel * 1.5)}%` }}
                  />
                </div>
              </div>
            )}

            {!isRecording ? (
              <button
                onClick={startRecording}
                className="h-16 w-16 rounded-full bg-rose-600 hover:bg-rose-500 transition-all shadow-lg shadow-rose-600/30 flex items-center justify-center text-white group"
              >
                <Mic className="w-7 h-7 group-hover:scale-110 transition-transform" />
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="h-16 w-16 rounded-full bg-rose-500 animate-pulse transition-all shadow-lg shadow-rose-500/40 flex items-center justify-center text-white"
              >
                <Square className="w-6 h-6" />
              </button>
            )}

            <p className="text-xs text-slate-400 mt-3">
              {isRecording ? 'Click square to finish recording' : 'Click microphone to record 10-30 seconds of clean speech or singing'}
            </p>
          </div>
        ) : (
          <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500/60 transition-colors rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-slate-950/40 text-center">
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a"
              onChange={handleFileUpload}
              className="hidden"
              id="voice-file-input"
            />
            <label htmlFor="voice-file-input" className="cursor-pointer flex flex-col items-center">
              <div className="h-12 w-12 rounded-full bg-slate-800 hover:bg-indigo-600/20 transition-colors flex items-center justify-center mb-3 text-slate-400">
                <Upload className="w-6 h-6" />
              </div>
              <span className="text-sm font-medium text-slate-200">Upload voice sample</span>
              <span className="text-xs text-slate-400 mt-1">10–30s clean vocal file, no background music</span>
            </label>
          </div>
        )
      ) : (
        <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlayback}
              className="h-10 w-10 rounded-full bg-emerald-600 hover:bg-emerald-500 transition-colors flex items-center justify-center text-white shrink-0 shadow-md shadow-emerald-600/30"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div>
              <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Voice Sample Ready</span>
              </div>
              <p className="text-xs text-slate-400">Preview your recorded reference</p>
            </div>
            <audio
              ref={audioPreviewRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
            />
          </div>

          <button
            onClick={resetSample}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white bg-slate-700/60 hover:bg-slate-700 text-xs transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Retake</span>
          </button>
        </div>
      )}
    </div>
  );
}
