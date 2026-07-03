// Every server endpoint lives here — if the backend routes end up differing,
// this file is the only place to fix.
export const endpoints = {
  wsStream: '/ws/stream',
  wsLogs: '/ws/logs',
  replay: '/api/replay',
} as const;

// Same-origin by default (FastAPI serves the build; Vite dev proxies /ws + /api).
// Override with VITE_DAIYA_SERVER when pointing elsewhere.
const serverBase = (import.meta.env.VITE_DAIYA_SERVER as string | undefined) || window.location.origin;

export function httpUrl(path: string): string {
  return new URL(path, serverBase).toString();
}

export function wsUrl(path: string, params?: Record<string, string | number>): string {
  const url = new URL(path, serverBase);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  if (params) {
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, String(v));
  }
  return url.toString();
}

// ---- wire protocol -------------------------------------------------------

export type SourceId = 'browser-mic' | 'server-mic' | 'desktop' | 'replay';

export interface Word {
  word: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface Segment {
  segment_id: string;
  speaker: string | null;
  start: number;
  end: number;
  text: string;
  final: boolean;
  confidence?: number;
  words?: Word[];
}

export type SegmentPatch = Partial<Segment> & { segment_id: string };

export type ServerEvent =
  | { kind: 'segment'; phase: 'partial' | 'final' | 'update'; patch: SegmentPatch }
  | { kind: 'log'; level: string; message: string }
  | { kind: 'error'; message: string };

/** Parse one JSON text message from /ws/stream or /ws/logs. Tolerates flat or `data`-nested payloads. */
export function parseEvent(raw: string): ServerEvent | null {
  let msg: Record<string, unknown>;
  try {
    msg = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return { kind: 'log', level: 'info', message: raw }; // plain-text log line
  }
  if (typeof msg !== 'object' || msg === null) return null;

  const type = String(msg.type ?? msg.event ?? '');
  const d = (msg.data && typeof msg.data === 'object' ? msg.data : msg) as Record<string, unknown>;

  if (type === 'transcript.partial' || type === 'transcript.final' || type === 'transcript.update') {
    if (d.segment_id === undefined || d.segment_id === null) return null;
    const patch: SegmentPatch = { segment_id: String(d.segment_id) };
    if ('speaker' in d) patch.speaker = d.speaker === null ? null : String(d.speaker);
    if (typeof d.start === 'number') patch.start = d.start;
    if (typeof d.end === 'number') patch.end = d.end;
    if (typeof d.text === 'string') patch.text = d.text;
    if (typeof d.final === 'boolean') patch.final = d.final;
    if (typeof d.confidence === 'number') patch.confidence = d.confidence;
    if (Array.isArray(d.words)) patch.words = d.words as Word[];
    return { kind: 'segment', phase: type.slice('transcript.'.length) as 'partial' | 'final' | 'update', patch };
  }
  if (type === 'log') {
    return {
      kind: 'log',
      level: String(d.level ?? 'info').toLowerCase(),
      message: String(d.message ?? raw),
    };
  }
  if (type === 'error') {
    return { kind: 'error', message: String(d.message ?? 'Unknown server error') };
  }
  return null;
}

// ---- tuning settings -----------------------------------------------------

export interface Settings {
  vad_threshold: number;
  utterance_cap_seconds: number;
  window_seconds: number;
  hop_seconds: number;
  commit_delay_seconds: number;
  match_threshold: number;
}

// Defaults mirror the lab's RealtimeDiarizationConfig "balanced" profile
// and MATCH_THRESHOLD=0.38 from lab/statefull-diarization.
export const DEFAULT_SETTINGS: Settings = {
  vad_threshold: 0.012,
  utterance_cap_seconds: 8,
  window_seconds: 10,
  hop_seconds: 1,
  commit_delay_seconds: 5,
  match_threshold: 0.38,
};

export interface SettingField {
  key: keyof Settings;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  hint: string;
}

export const SETTING_FIELDS: SettingField[] = [
  {
    key: 'vad_threshold',
    label: 'VAD threshold',
    min: 0.001,
    max: 0.05,
    step: 0.001,
    hint: 'Energy threshold needed to open an utterance',
  },
  {
    key: 'utterance_cap_seconds',
    label: 'Utterance cap',
    min: 2,
    max: 20,
    step: 0.5,
    unit: 's',
    hint: 'Force a transcription pass after this long',
  },
  {
    key: 'window_seconds',
    label: 'Diarization window',
    min: 3,
    max: 20,
    step: 0.5,
    unit: 's',
    hint: 'Rolling audio window fed to the diarizer',
  },
  {
    key: 'hop_seconds',
    label: 'Diarization hop',
    min: 0.25,
    max: 4,
    step: 0.25,
    unit: 's',
    hint: 'How often the window advances',
  },
  {
    key: 'commit_delay_seconds',
    label: 'Commit delay',
    min: 1,
    max: 12,
    step: 0.5,
    unit: 's',
    hint: 'How long a turn stays provisional before committing',
  },
  {
    key: 'match_threshold',
    label: 'Match threshold',
    min: 0.1,
    max: 0.9,
    step: 0.01,
    hint: 'Speaker-embedding distance for re-identification',
  },
];
