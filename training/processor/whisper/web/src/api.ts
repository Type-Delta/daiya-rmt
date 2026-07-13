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
}

export interface Job {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  createdAt: string;
  finishedAt: string | null;
  commands: string[][];
  outputs: Record<string, string>;
  log: string[];
}

export interface Session {
  id: string;
  directory: string;
  reviewer: string;
}

export interface Review {
  human: { action: 'confirmed' | 'edited' | 'skipped'; label: string };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const data = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(data.error ?? `Request failed (${response.status}).`);
  return data;
}

export const api = {
  jobs: () => request<{ jobs: Job[] }>('/api/jobs'),
  startAutoLabel: (payload: Record<string, unknown>) =>
    request<{ job: Job }>('/api/jobs/autolabel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  startValidation: (payload: Record<string, unknown>) =>
    request<{ job: Job }>('/api/jobs/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  loadDataset: (payload: Record<string, unknown>) =>
    request<{ rows: LabelRow[]; session: Session }>('/api/dataset/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  saveReview: (payload: Record<string, unknown>) =>
    request<{ review: Review }>('/api/review/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
};

export function audioUrl(path: string): string {
  return `/api/audio?path=${encodeURIComponent(path)}`;
}
