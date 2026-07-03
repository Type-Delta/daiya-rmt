import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CaretDown,
  CaretUp,
  FileAudio,
  HardDrives,
  Microphone,
  Monitor,
  Play,
  SlidersHorizontal,
  Stop,
  TerminalWindow,
  Trash,
  UploadSimple,
  WarningCircle,
  Waveform,
  X,
} from '@phosphor-icons/react';
import {
  DEFAULT_SETTINGS,
  ENGINE_TOGGLES,
  SETTING_FIELDS,
  endpoints,
  httpUrl,
  parseEvent,
  wsUrl,
  type Segment,
  type SegmentPatch,
  type Settings,
  type SourceId,
} from './api';
import { startMic, type MicHandle } from './audio/mic';

type Status = 'idle' | 'connecting' | 'live' | 'error';

const SOURCES: { id: SourceId; label: string; icon: typeof Microphone }[] = [
  { id: 'browser-mic', label: 'Browser mic', icon: Microphone },
  { id: 'server-mic', label: 'Server mic', icon: HardDrives },
  { id: 'desktop', label: 'Desktop audio', icon: Monitor },
  { id: 'replay', label: 'Replay file', icon: FileAudio },
];

// Categorical speaker hues at matched lightness/chroma; assigned by first appearance.
const SPEAKER_COLORS = [
  'oklch(0.78 0.12 200)',
  'oklch(0.78 0.13 25)',
  'oklch(0.80 0.13 150)',
  'oklch(0.78 0.12 300)',
  'oklch(0.82 0.12 65)',
  'oklch(0.76 0.12 255)',
  'oklch(0.80 0.12 340)',
  'oklch(0.80 0.11 180)',
];

const FOCUS_RING =
  'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent';

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec < 10 ? '0' : ''}${sec.toFixed(1)}`;
}

// ---- transcript state ------------------------------------------------------

function upsertSegment(prev: Segment[], phase: 'partial' | 'final' | 'update', patch: SegmentPatch): Segment[] {
  const i = prev.findIndex((s) => s.segment_id === patch.segment_id);
  const base: Segment =
    i >= 0
      ? prev[i]
      : { segment_id: patch.segment_id, speaker: null, start: 0, end: 0, text: '', final: false };
  const merged: Segment = {
    ...base,
    ...patch,
    final: phase === 'final' ? true : (patch.final ?? base.final),
  };
  const next = i >= 0 ? prev.map((s, j) => (j === i ? merged : s)) : [...prev, merged];
  // Full re-sort per event is fine for session-sized transcripts; switch to
  // sorted insertion if a profiler ever shows this path matters.
  return next.sort((a, b) => a.start - b.start || a.segment_id.localeCompare(b.segment_id));
}

// ---- console (server log) state ---------------------------------------------

interface LogLine {
  id: number;
  ts: string;
  level: string;
  text: string;
}

let nextLogId = 0;

function useConsole(tailEnabled: boolean) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const push = useCallback((level: string, text: string) => {
    const line: LogLine = { id: nextLogId++, ts: new Date().toLocaleTimeString(), level, text };
    setLines((prev) => (prev.length >= 500 ? [...prev.slice(prev.length - 499), line] : [...prev, line]));
  }, []);

  // Tail /ws/logs only while the panel is open; reconnect while it stays open.
  useEffect(() => {
    if (!tailEnabled) return;
    let ws: WebSocket | null = null;
    let timer = 0;
    let disposed = false;
    const connect = () => {
      ws = new WebSocket(wsUrl(endpoints.wsLogs));
      ws.onmessage = (e) => {
        if (typeof e.data !== 'string') return;
        const ev = parseEvent(e.data);
        if (ev?.kind === 'log') push(ev.level, ev.message);
        else if (ev?.kind === 'error') push('error', ev.message);
      };
      ws.onclose = () => {
        if (!disposed) timer = window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, [tailEnabled, push]);

  const clear = useCallback(() => setLines([]), []);
  return { lines, push, clear };
}

// ---- streaming session -------------------------------------------------------

function useSession(pushLog: (level: string, text: string) => void) {
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [clock, setClock] = useState(0);
  const [diarOnly, setDiarOnly] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicHandle | null>(null);
  const replayAbortRef = useRef<AbortController | null>(null);
  const closingRef = useRef(false);

  const stop = useCallback(() => {
    closingRef.current = true;
    replayAbortRef.current?.abort();
    replayAbortRef.current = null;
    micRef.current?.stop();
    micRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('idle');
  }, []);

  const applyServerEvent = useCallback(
    (raw: string) => {
      const ev = parseEvent(raw);
      if (!ev) return;
      if (ev.kind === 'segment') setSegments((prev) => upsertSegment(prev, ev.phase, ev.patch));
      else if (ev.kind === 'tick') setClock(ev.time);
      else if (ev.kind === 'log') pushLog(ev.level, ev.message);
      else setError(ev.message);
    },
    [pushLog],
  );

  const startReplay = useCallback(
    async (settings: Settings, file: File) => {
      const controller = new AbortController();
      replayAbortRef.current = controller;

      const fail = (message: string) => {
        if (closingRef.current) return;
        setError(message);
        setStatus('error');
        closingRef.current = true;
        replayAbortRef.current?.abort();
        replayAbortRef.current = null;
      };

      try {
        const fd = new FormData();
        fd.append('file', file);
        fd.append('pace', 'false');
        fd.append('chunk_seconds', '0.5');
        fd.append('config', JSON.stringify(settings));

        const res = await fetch(httpUrl(endpoints.replay), {
          method: 'POST',
          body: fd,
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Replay upload failed (HTTP ${res.status}).`);
        if (!res.body) throw new Error('Replay response did not include a stream.');

        setStatus('live');

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });

          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() ?? '';
          for (const line of lines) {
            if (line.trim()) applyServerEvent(line);
          }

          if (done) break;
        }

        const tail = buffer.trim();
        if (tail) applyServerEvent(tail);
        if (!closingRef.current) setStatus('idle');
      } catch (err) {
        if (controller.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) return;
        fail(err instanceof Error ? err.message : String(err));
      } finally {
        if (replayAbortRef.current === controller) replayAbortRef.current = null;
      }
    },
    [applyServerEvent],
  );

  const start = useCallback(
    (source: SourceId, settings: Settings, file: File | null) => {
      setError(null);
      setSegments([]);
      setClock(0);
      setDiarOnly(!settings.enable_asr && settings.enable_diarization);
      setStatus('connecting');
      closingRef.current = false;
      replayAbortRef.current?.abort();
      replayAbortRef.current = null;

      if (source === 'replay') {
        if (!file) {
          setError('Pick an audio file to replay first.');
          setStatus('error');
          closingRef.current = true;
          return;
        }
        void startReplay(settings, file);
        return;
      }

      const ws = new WebSocket(wsUrl(endpoints.wsStream, { source, ...settings }));
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      const fail = (message: string) => {
        setError(message);
        setStatus('error');
        closingRef.current = true;
        micRef.current?.stop();
        micRef.current = null;
        if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) ws.close();
      };

      ws.onopen = async () => {
        ws.send(JSON.stringify({ type: 'config', settings }));
        ws.send(JSON.stringify({ type: 'source', source }));
        try {
          if (source === 'browser-mic') {
            micRef.current = await startMic((frame) => {
              if (ws.readyState === WebSocket.OPEN) ws.send(frame);
            });
          }
          setStatus('live');
        } catch (err) {
          fail(err instanceof Error ? err.message : String(err));
        }
      };
      ws.onmessage = (e) => {
        if (typeof e.data !== 'string') return;
        applyServerEvent(e.data);
      };
      ws.onerror = () => {
        if (!closingRef.current) fail('Stream connection failed — is the Daiya server running?');
      };
      ws.onclose = () => {
        if (closingRef.current) return;
        micRef.current?.stop();
        micRef.current = null;
        setStatus((s) => (s === 'error' ? s : 'idle'));
      };
    },
    [applyServerEvent, startReplay],
  );

  /** Push new settings into the running session; false when not connected. */
  const pushSettings = useCallback((settings: Settings) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'config', settings }));
      return true;
    }
    return false;
  }, []);

  return { status, error, dismissError: () => setError(null), segments, clock, diarOnly, start, stop, pushSettings };
}

// ---- shared scroll behavior ---------------------------------------------------

function useStickToBottom(dep: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const onScroll = useCallback(() => {
    const el = ref.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [dep]);
  return { ref, onScroll };
}

// ---- components -----------------------------------------------------------------

function StatusPill({ status }: { status: Status }) {
  const cfg = {
    idle: { dot: 'bg-faint', label: 'Idle' },
    connecting: { dot: 'bg-warn animate-pulse', label: 'Connecting…' },
    live: { dot: 'bg-ok', label: 'Live' },
    error: { dot: 'bg-danger', label: 'Error' },
  }[status];
  return (
    <span
      aria-live="polite"
      className="flex items-center gap-1.5 rounded-full border border-edge bg-surface px-2.5 py-1 text-xs text-muted"
    >
      <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
}

function SourcePicker({
  value,
  onChange,
  disabled,
}: {
  value: SourceId;
  onChange: (id: SourceId) => void;
  disabled: boolean;
}) {
  return (
    <div role="radiogroup" aria-label="Audio source" className="flex flex-wrap gap-1 rounded-lg bg-surface p-1">
      {SOURCES.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={value === id}
          disabled={disabled}
          onClick={() => onChange(id)}
          className={`flex h-8 items-center gap-1.5 rounded-md px-2.5 text-sm transition-colors duration-150 ${FOCUS_RING} ${
            value === id ? 'bg-raised text-ink' : 'text-muted hover:text-ink'
          } ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}
        >
          <Icon size={16} aria-hidden />
          {label}
        </button>
      ))}
    </div>
  );
}

function SettingsPanel({
  settings,
  onChange,
  onApply,
  applied,
  live,
  onClose,
}: {
  settings: Settings;
  onChange: (next: Settings) => void;
  onApply: () => void;
  applied: boolean;
  live: boolean;
  onClose?: () => void;
}) {
  return (
    <div className="space-y-5 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Tuning</h2>
        {onClose && (
          <button
            type="button"
            aria-label="Close settings"
            onClick={onClose}
            className={`rounded p-1 text-muted hover:text-ink ${FOCUS_RING}`}
          >
            <X size={16} aria-hidden />
          </button>
        )}
      </div>
      <div className="space-y-3 border-b border-edge pb-4">
        {ENGINE_TOGGLES.map((t) => (
          <div key={t.key} className="space-y-1">
            <label className="flex cursor-pointer items-center justify-between gap-2 text-sm">
              {t.label}
              <input
                type="checkbox"
                checked={settings[t.key]}
                onChange={(e) => onChange({ ...settings, [t.key]: e.target.checked })}
                className={`h-4 w-4 accent-[var(--primary)] ${FOCUS_RING}`}
              />
            </label>
            <p className="text-xs text-faint">{t.hint}</p>
          </div>
        ))}
        <p className="text-xs text-faint">Engine toggles take effect on the next Start.</p>
      </div>
      {SETTING_FIELDS.map((f) => {
        const value = settings[f.key];
        const set = (n: number) => onChange({ ...settings, [f.key]: n });
        return (
          <div key={f.key} className="space-y-1.5">
            <div className="flex items-baseline justify-between gap-2">
              <label htmlFor={f.key} className="text-sm">
                {f.label}
              </label>
              <span className="text-xs tabular-nums text-muted">
                {value}
                {f.unit ?? ''}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <input
                id={f.key}
                type="range"
                min={f.min}
                max={f.max}
                step={f.step}
                value={value}
                onChange={(e) => set(Number(e.target.value))}
                className="h-4 flex-1"
              />
              <input
                type="number"
                aria-label={`${f.label} value`}
                min={f.min}
                max={f.max}
                step={f.step}
                value={value}
                onChange={(e) => {
                  const n = Number.parseFloat(e.target.value);
                  if (Number.isFinite(n)) set(n);
                }}
                className={`w-20 rounded border border-edge bg-raised px-2 py-1 text-right text-sm tabular-nums ${FOCUS_RING}`}
              />
            </div>
            <p className="text-xs text-faint">{f.hint}</p>
          </div>
        );
      })}
      <div className="space-y-2 border-t border-edge pt-4">
        <button
          type="button"
          onClick={onApply}
          disabled={!live}
          className={`h-9 w-full rounded-md border border-edge bg-raised text-sm font-medium text-ink transition-colors duration-150 hover:border-primary disabled:cursor-not-allowed disabled:opacity-50 ${FOCUS_RING}`}
        >
          {applied ? 'Sent ✓' : 'Apply to live session'}
        </button>
        <p className="text-xs text-faint">
          Settings ride along when you connect; this button pushes them into the running session.
        </p>
      </div>
    </div>
  );
}

function Transcript({ segments, status }: { segments: Segment[]; status: Status }) {
  const colorOf = useMemo(() => {
    const map = new Map<string, string>();
    for (const s of segments) {
      if (s.speaker && !map.has(s.speaker)) {
        map.set(s.speaker, SPEAKER_COLORS[map.size % SPEAKER_COLORS.length]);
      }
    }
    return (speaker: string | null) => (speaker && map.get(speaker)) || 'var(--muted)';
  }, [segments]);
  const { ref, onScroll } = useStickToBottom(segments);

  if (segments.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <Waveform size={40} className="text-faint" aria-hidden />
        <p className="font-medium">{status === 'live' ? 'Listening…' : 'No transcript yet'}</p>
        <p className="max-w-sm text-sm text-muted">
          Pick a source and press Start. Dim text is a provisional pass; it turns solid once the
          diarizer commits the speaker.
        </p>
      </div>
    );
  }

  return (
    <div ref={ref} onScroll={onScroll} className="h-full overflow-y-auto px-4 py-4 md:px-8">
      <div className="mx-auto max-w-2xl space-y-0.5">
        {segments.map((s, i) => {
          const showSpeaker = i === 0 || segments[i - 1].speaker !== s.speaker;
          const color = colorOf(s.speaker);
          return (
            <div key={s.segment_id} className={showSpeaker ? 'pt-4 first:pt-0' : ''}>
              {showSpeaker && (
                <div className="mb-1 flex items-center gap-2">
                  <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: color }} />
                  <span className="text-xs font-semibold" style={{ color }}>
                    {s.speaker ?? 'Unknown speaker'}
                  </span>
                  <span className="text-xs tabular-nums text-faint">{fmtTime(s.start)}</span>
                </div>
              )}
              <p
                className={`leading-relaxed transition-opacity duration-200 ${
                  s.final ? 'opacity-100' : 'opacity-60'
                }`}
              >
                {s.text || '…'}
                {s.confidence !== undefined && (
                  <span className="ml-2 text-xs tabular-nums text-faint">
                    {Math.round(s.confidence * 100)}%
                  </span>
                )}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- diarization-only timeline ---------------------------------------------

// Turns closer together than this merge into one line; longer gaps become a
// "No speaker" line. Matches the diarizer's ~1 s hop granularity.
const RUN_GAP_SECONDS = 1.0;

interface SpeakerRun {
  speaker: string | null; // null = silence / no speaker detected
  start: number;
  end: number;
}

/** One line per speaker change; silence gaps (and the live clock) become "No speaker" lines. */
function buildRuns(segments: Segment[], clock: number): SpeakerRun[] {
  const runs: SpeakerRun[] = [];
  for (const s of segments) {
    const last = runs[runs.length - 1];
    if (last && last.speaker === s.speaker && s.start - last.end <= RUN_GAP_SECONDS) {
      last.end = Math.max(last.end, s.end);
      continue;
    }
    if (last && s.start - last.end > RUN_GAP_SECONDS) {
      runs.push({ speaker: null, start: last.end, end: s.start });
    }
    runs.push({ speaker: s.speaker, start: s.start, end: s.end });
  }
  const last = runs[runs.length - 1];
  if (!last) {
    if (clock > 0) runs.push({ speaker: null, start: 0, end: clock });
  } else if (clock - last.end > RUN_GAP_SECONDS) {
    runs.push({ speaker: null, start: last.end, end: clock });
  }
  return runs;
}

function SpeakerTimeline({ segments, clock, status }: { segments: Segment[]; clock: number; status: Status }) {
  const runs = useMemo(() => buildRuns(segments, clock), [segments, clock]);
  const colorOf = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of runs) {
      if (r.speaker && !map.has(r.speaker)) {
        map.set(r.speaker, SPEAKER_COLORS[map.size % SPEAKER_COLORS.length]);
      }
    }
    return (speaker: string | null) => (speaker && map.get(speaker)) || 'var(--faint)';
  }, [runs]);
  const { ref, onScroll } = useStickToBottom(runs);

  if (runs.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <Waveform size={40} className="text-faint" aria-hidden />
        <p className="font-medium">{status === 'live' ? 'Listening…' : 'Speaker timeline'}</p>
        <p className="max-w-sm text-sm text-muted">
          ASR is off — this session shows who is speaking and when, with no transcript text.
        </p>
      </div>
    );
  }

  return (
    <div ref={ref} onScroll={onScroll} className="h-full overflow-y-auto px-4 py-4 md:px-8">
      <div className="mx-auto max-w-2xl">
        {runs.map((r, i) => {
          const live = status === 'live' && i === runs.length - 1;
          const color = colorOf(r.speaker);
          return (
            <div key={i} className="flex items-center gap-3 py-1.5">
              <span
                aria-hidden
                className={`h-2 w-2 shrink-0 rounded-full ${live ? 'animate-pulse' : ''}`}
                style={{ background: color }}
              />
              <span className={`text-sm font-semibold ${r.speaker ? '' : 'font-normal text-faint'}`} style={r.speaker ? { color } : undefined}>
                {r.speaker ?? 'No speaker'}
              </span>
              <span className="text-xs tabular-nums text-muted">
                {fmtTime(r.start)} – {fmtTime(r.end)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function levelClass(level: string): string {
  if (level === 'error' || level === 'critical') return 'text-danger';
  if (level === 'warning' || level === 'warn') return 'text-warn';
  if (level === 'debug') return 'text-faint';
  return 'text-accent';
}

function ConsoleDrawer({
  open,
  onToggle,
  lines,
  onClear,
}: {
  open: boolean;
  onToggle: () => void;
  lines: LogLine[];
  onClear: () => void;
}) {
  const { ref, onScroll } = useStickToBottom(lines);
  return (
    <section className="border-t border-edge bg-surface">
      <div className="flex h-9 items-center gap-3 px-3">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className={`flex items-center gap-2 rounded text-sm text-muted hover:text-ink ${FOCUS_RING}`}
        >
          <TerminalWindow size={16} aria-hidden />
          Server console
          {open ? <CaretDown size={14} aria-hidden /> : <CaretUp size={14} aria-hidden />}
        </button>
        {lines.length > 0 && <span className="text-xs tabular-nums text-faint">{lines.length}</span>}
        <div className="flex-1" />
        {open && lines.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            aria-label="Clear console"
            className={`rounded p-1 text-faint hover:text-ink ${FOCUS_RING}`}
          >
            <Trash size={15} aria-hidden />
          </button>
        )}
      </div>
      {open && (
        <div
          ref={ref}
          onScroll={onScroll}
          className="h-44 overflow-y-auto border-t border-edge px-3 py-2 font-mono text-xs md:h-56"
        >
          {lines.length === 0 ? (
            <p className="text-faint">Waiting for the server’s log stream ({endpoints.wsLogs})…</p>
          ) : (
            lines.map((l) => (
              <div key={l.id} className="flex gap-2 whitespace-pre-wrap break-words leading-5">
                <span className="shrink-0 tabular-nums text-faint">{l.ts}</span>
                <span className={`w-14 shrink-0 uppercase ${levelClass(l.level)}`}>{l.level}</span>
                <span className="min-w-0 text-muted">{l.text}</span>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
}

// ---- app ----------------------------------------------------------------------

export default function App() {
  const [source, setSource] = useState<SourceId>('browser-mic');
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [file, setFile] = useState<File | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [applied, setApplied] = useState(false);

  const { lines, push, clear } = useConsole(consoleOpen);
  const { status, error, dismissError, segments, clock, diarOnly, start, stop, pushSettings } = useSession(push);
  const running = status === 'live' || status === 'connecting';

  const applyLive = () => {
    if (pushSettings(settings)) {
      setApplied(true);
      window.setTimeout(() => setApplied(false), 1500);
    }
  };

  useEffect(() => {
    if (!settingsOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setSettingsOpen(false);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [settingsOpen]);

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-edge px-4 py-2.5 md:px-6">
        <div className="flex items-center gap-2">
          <Waveform size={20} weight="bold" className="text-primary" aria-hidden />
          <h1 className="text-sm font-semibold tracking-tight">Daiya</h1>
          <span className="text-xs text-faint">v0 testbed</span>
        </div>
        <StatusPill status={status} />
        <div className="flex-1" />
        <button
          type="button"
          aria-label="Tuning settings"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((v) => !v)}
          className={`rounded-md p-2 text-muted hover:text-ink lg:hidden ${FOCUS_RING}`}
        >
          <SlidersHorizontal size={18} aria-hidden />
        </button>
      </header>

      {error && (
        <div
          role="alert"
          className="flex items-center gap-2 border-b border-edge px-4 py-2 text-sm md:px-6"
          style={{ background: 'color-mix(in oklab, var(--danger) 12%, var(--bg))' }}
        >
          <WarningCircle size={16} className="shrink-0 text-danger" aria-hidden />
          <span className="min-w-0 flex-1">{error}</span>
          <button
            type="button"
            aria-label="Dismiss error"
            onClick={dismissError}
            className={`rounded p-1 text-muted hover:text-ink ${FOCUS_RING}`}
          >
            <X size={14} aria-hidden />
          </button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-b border-edge px-4 py-2 md:px-6">
        <SourcePicker value={source} onChange={setSource} disabled={running} />
        {source === 'replay' && (
          <label
            className={`flex h-9 cursor-pointer items-center gap-2 rounded-md border border-edge bg-surface px-3 text-sm text-muted hover:text-ink has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent ${
              running ? 'cursor-not-allowed opacity-50' : ''
            }`}
          >
            <UploadSimple size={16} aria-hidden />
            <span className="max-w-40 truncate">{file ? file.name : 'Choose audio file'}</span>
            <input
              type="file"
              accept="audio/*,.wav,.flac,.mp3,.m4a,.ogg"
              className="sr-only"
              disabled={running}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
        )}
        <div className="flex-1" />
        {running ? (
          <button
            type="button"
            onClick={stop}
            className={`flex h-9 items-center gap-2 rounded-md border border-edge bg-raised px-4 text-sm font-semibold text-ink transition-colors duration-150 hover:border-danger hover:text-danger ${FOCUS_RING}`}
          >
            <Stop size={15} weight="fill" aria-hidden />
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => start(source, settings, file)}
            disabled={(source === 'replay' && !file) || (!settings.enable_asr && !settings.enable_diarization)}
            title={
              !settings.enable_asr && !settings.enable_diarization
                ? 'Enable ASR or diarization first'
                : source === 'replay' && !file
                  ? 'Choose a file first'
                  : undefined
            }
            className={`flex h-9 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-ink transition-opacity duration-150 hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 ${FOCUS_RING}`}
          >
            <Play size={15} weight="fill" aria-hidden />
            Start
          </button>
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        <main className="min-w-0 flex-1">
          {diarOnly ? (
            <SpeakerTimeline segments={segments} clock={clock} status={status} />
          ) : (
            <Transcript segments={segments} status={status} />
          )}
        </main>
        <aside className="hidden w-80 shrink-0 overflow-y-auto border-l border-edge bg-surface lg:block">
          <SettingsPanel
            settings={settings}
            onChange={setSettings}
            onApply={applyLive}
            applied={applied}
            live={status === 'live'}
          />
        </aside>
      </div>

      {settingsOpen && (
        <div className="fixed inset-0 z-30 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setSettingsOpen(false)} />
          <div className="absolute inset-y-0 right-0 w-80 max-w-[85vw] overflow-y-auto border-l border-edge bg-surface">
            <SettingsPanel
              settings={settings}
              onChange={setSettings}
              onApply={applyLive}
              applied={applied}
              live={status === 'live'}
              onClose={() => setSettingsOpen(false)}
            />
          </div>
        </div>
      )}

      <ConsoleDrawer open={consoleOpen} onToggle={() => setConsoleOpen((v) => !v)} lines={lines} onClear={clear} />
    </div>
  );
}
