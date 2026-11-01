export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

const DEFAULT_HEADERS = {
  'ngrok-skip-browser-warning': 'true',
};

export interface HealthResponse {
  status: string;
  gpu_available: boolean;
  gpu_name: string;
  active_jobs: number;
}

export interface VoiceModelInfo {
  id: string;
  name: string;
  type: 'zero-shot' | 'rvc-trained';
  description: string;
  recommended_shift?: number;
}

export interface JobOptions {
  semitone_shift: number;
  diffusion_steps: number;
  output_format: 'mp3' | 'wav';
  rvc_index_rate?: number;
}

export interface JobCreateResponse {
  job_id: string;
  status: string;
  upload_song_url: string;
  upload_sample_url: string;
}

export interface JobStatusResponse {
  job_id: string;
  engine: 'seed-vc' | 'rvc';
  status: 'CREATED' | 'UPLOADED' | 'QUEUED' | 'VALIDATING' | 'SEPARATING' | 'CONVERTING' | 'MIXING' | 'DONE' | 'FAILED';
  progress: number;
  message: string;
  elapsed_seconds: number;
  error?: string | null;
  outputs?: {
    final_mix_url: string;
    vocals_url: string;
    original_song_url: string;
  } | null;
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, {
    method: 'GET',
    headers: DEFAULT_HEADERS,
  });
  if (!res.ok) {
    throw new Error(`Backend unreachable: HTTP ${res.status}`);
  }
  return res.json();
}

export async function getAvailableModels(): Promise<VoiceModelInfo[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/models`, {
      method: 'GET',
      headers: DEFAULT_HEADERS,
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.models || [];
  } catch (e) {
    return [
      { id: 'zero-shot', name: 'Your Voice (Instant Zero-Shot)', type: 'zero-shot', description: 'Requires 15s voice recording. Instant 0 wait time.' },
      { id: 'rvc-studio', name: 'Custom Studio Voice (RVC v2)', type: 'rvc-trained', description: 'Trained RVC v2 model with 95%+ studio accuracy.' },
    ];
  }
}

export async function createConversionJob(
  options: JobOptions,
  engine: 'seed-vc' | 'rvc' = 'seed-vc',
  modelId: string = 'zero-shot'
): Promise<JobCreateResponse> {
  const res = await fetch(`${API_BASE_URL}/api/jobs`, {
    method: 'POST',
    headers: {
      ...DEFAULT_HEADERS,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      engine,
      model_id: modelId,
      options,
    }),
  });
  if (!res.ok) {
    throw new Error(`Failed to create job: HTTP ${res.status}`);
  }
  return res.json();
}

export async function uploadAudioFile(
  jobId: string,
  fileType: 'song' | 'sample',
  file: File | Blob,
  filename?: string
): Promise<{ status: string }> {
  const formData = new FormData();
  formData.append('file', file, filename || `${fileType}.wav`);

  const res = await fetch(`${API_BASE_URL}/api/jobs/${jobId}/upload/${fileType}`, {
    method: 'POST',
    headers: DEFAULT_HEADERS,
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Upload failed for ${fileType}: HTTP ${res.status}`);
  }
  return res.json();
}

export async function startConversion(jobId: string, consent: boolean = true): Promise<{ status: string }> {
  const formData = new FormData();
  formData.append('consent', consent ? 'true' : 'false');

  const res = await fetch(`${API_BASE_URL}/api/jobs/${jobId}/start`, {
    method: 'POST',
    headers: DEFAULT_HEADERS,
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to start conversion: HTTP ${res.status}`);
  }
  return res.json();
}

export async function pollJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`, {
    method: 'GET',
    headers: DEFAULT_HEADERS,
  });
  if (!res.ok) {
    throw new Error(`Failed to check job status: HTTP ${res.status}`);
  }
  return res.json();
}

export async function deleteJobData(jobId: string): Promise<{ deleted: boolean }> {
  const res = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`, {
    method: 'DELETE',
    headers: DEFAULT_HEADERS,
  });
  if (!res.ok) {
    throw new Error(`Failed to delete job data: HTTP ${res.status}`);
  }
  return res.json();
}
