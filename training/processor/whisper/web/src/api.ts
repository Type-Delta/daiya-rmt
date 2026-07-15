export type Disposition = 'keep' | 'review' | 'correct' | 'drop' | 'unclassified' | string;

export interface Evidence {
  name: string;
  value: string | number | boolean | null;
  source?: string;
  detail?: string | null;
}

export interface LabelRow {
  id: string;
  index: number;
  sourceUri: string;
  audioPath: string | null;
  audioAvailable: boolean;
  disposition: Disposition;
  originalLabel: string;
  proposedLabel: string | null;
  language: string;
  duration: number | null;
  reasons: string[];
  evidence: Evidence[];
  sourceStart: number | null;
  sourceEnd: number | null;
  reviewed: boolean;
}

export interface Job {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  createdAt: string;
  finishedAt: string | null;
  commands: string[][];
  outputs: Record<string, string>;
  log: string[];
  cancelRequested: boolean;
  progress: { current: number; total: number; fraction: number; detail: string };
}

export interface Session {
  id: string;
  directory: string;
  reviewer: string;
  resumed: boolean;
}

export interface Review {
  human: { action: 'confirmed' | 'edited' | 'skipped'; label: string };
}
export interface ImportSummary { imported: number; matched: number; new: number; unchanged: number; conflicts: number; unmatched: number; }

export interface WorkbenchConfig {
  projectRoot: string;
  autoLabel: { inputDir: string; outputDir: string; workDir: string; noOverlapFilter: boolean };
  validation: {
    metadataPath: string; audioRoot: string; outputRoot: string; datasetVersion: string; thaiEngine: string;
    expectedScripts: string; reviewThreshold: string; minIssues: string; allowlist: string;
    japaneseDictionary: string; englishDictionary: string;
  };
  review: { metadataPath: string; manifestPath: string; audioRoot: string; reviewRoot: string; reviewer: string };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const data = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(data.error ?? `Request failed (${response.status}).`);
  return data;
}

export const api = {
  configuration: () => request<WorkbenchConfig>('/api/config'),
  jobs: () => request<{ jobs: Job[] }>('/api/jobs'),
  cancelJob: (jobId: string) => request<{ job: Job }>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }),
  startAutoLabel: (payload: Record<string, unknown>) =>
    request<{ job: Job }>('/api/jobs/autolabel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  startValidation: (payload: Record<string, unknown>) =>
    request<{ job: Job }>('/api/jobs/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  loadDataset: (payload: Record<string, unknown>) =>
    request<{ rows: LabelRow[]; session: Session; reviews: Record<string, Review['human']> }>('/api/dataset/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  saveReview: (payload: Record<string, unknown>) =>
    request<{ review: Review; reviews?: Record<string, Review['human']> }>('/api/review/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  previewReviewImport: (payload: { sessionId: string; content: string }) =>
    request<{ summary: ImportSummary }>('/api/review/import/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  applyReviewImport: (payload: { sessionId: string; content: string; conflictPolicy: 'ours' | 'theirs' }) =>
    request<{ reviews: Record<string, Review['human']>; summary: ImportSummary }>('/api/review/import/apply', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  validatePath: (payload: { path: string; kind: 'file' | 'directory'; allowMissing?: boolean }) =>
    request<{ valid: boolean; exists: boolean; path?: string; message: string }>('/api/path/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  pickPath: (payload: { kind: 'file' | 'directory'; initialPath: string }) =>
    request<{ path: string | null }>('/api/path/pick', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
};

export function reviewExportUrl(sessionId: string): string {
  return `/api/review/export?sessionId=${encodeURIComponent(sessionId)}`;
}

export function audioUrl(path: string): string {
  return `/api/audio?path=${encodeURIComponent(path)}`;
}
