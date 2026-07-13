import { useEffect, useMemo, useState } from 'react';
import {
  ArrowClockwise,
  CaretLeft,
  CaretRight,
  Check,
  FileAudio,
  FloppyDisk,
  GearSix,
  ListBullets,
  MagnifyingGlass,
  Play,
  ShieldCheck,
  Sparkle,
  WarningCircle,
  X,
} from '@phosphor-icons/react';
import { api, audioUrl, type Job, type LabelRow, type Session } from './api';

type Filter = 'all' | 'keep' | 'review' | 'correct' | 'drop' | 'reviewed' | 'unreviewed';

interface ReviewState {
  action: 'confirmed' | 'edited' | 'skipped';
  label: string;
}

const FILTERS: { id: Filter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'review', label: 'Review' },
  { id: 'correct', label: 'Correct' },
  { id: 'keep', label: 'Keep' },
  { id: 'drop', label: 'Drop' },
  { id: 'unreviewed', label: 'Unreviewed' },
  { id: 'reviewed', label: 'Reviewed' },
];

function cleanPath(value: string): string {
  return value.trim();
}

function formatDuration(value: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Duration unknown';
  return `${value.toFixed(value < 10 ? 1 : 0)} s`;
}

function Status({ status }: { status: Job['status'] }) {
  const text = status === 'running' ? 'Running' : status === 'queued' ? 'Queued' : status === 'completed' ? 'Complete' : 'Failed';
  return <span className={`status status--${status}`}><span aria-hidden />{text}</span>;
}

function Disposition({ value }: { value: string }) {
  return <span className={`disposition disposition--${value}`}>{value}</span>;
}

function App() {
  const [rows, setRows] = useState<LabelRow[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [reviews, setReviews] = useState<Record<string, ReviewState>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [setupOpen, setSetupOpen] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [configured, setConfigured] = useState(false);
  const [auto, setAuto] = useState({ inputDir: '', outputDir: '', workDir: '', noOverlapFilter: false });
  const [validation, setValidation] = useState({ metadataPath: '', audioRoot: '', outputRoot: '', datasetVersion: '', thaiEngine: 'pn', expectedScripts: 'thai,latin', reviewThreshold: '0.2', minIssues: '1', allowlist: '', japaneseDictionary: '', englishDictionary: '' });
  const [load, setLoad] = useState({ metadataPath: '', manifestPath: '', audioRoot: '', reviewRoot: '', reviewer: '' });

  const refreshJobs = async () => {
    try {
      const data = await api.jobs();
      setJobs(data.jobs);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  useEffect(() => {
    const initialise = async () => {
      try {
        const [configuration, jobData] = await Promise.all([api.configuration(), api.jobs()]);
        setAuto(configuration.autoLabel);
        setValidation(configuration.validation);
        setLoad(configuration.review);
        setJobs(jobData.jobs);
        setConfigured(true);
        setNotice('Ready with project-root-relative configuration from the local server.');
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    };
    void initialise();
  }, []);
  useEffect(() => {
    if (!jobs.some((job) => job.status === 'queued' || job.status === 'running')) return;
    const id = window.setInterval(() => { void refreshJobs(); }, 2000);
    return () => window.clearInterval(id);
  }, [jobs]);

  const selected = rows.find((row) => row.id === selectedId) ?? null;
  const automaticText = selected ? (selected.proposedLabel ?? selected.originalLabel) : '';
  const reviewCount = Object.keys(reviews).length;
  const visibleRows = useMemo(() => rows.filter((row) => {
    const reviewed = Boolean(reviews[row.id]);
    const filterMatch = filter === 'all' || (filter === 'reviewed' && reviewed) || (filter === 'unreviewed' && !reviewed) || row.disposition === filter;
    const needle = query.trim().toLocaleLowerCase();
    const queryMatch = !needle || [row.sourceUri, row.originalLabel, row.proposedLabel ?? '', row.reasons.join(' ')].join(' ').toLocaleLowerCase().includes(needle);
    return filterMatch && queryMatch;
  }), [rows, reviews, filter, query]);

  useEffect(() => {
    if (visibleRows.length && !visibleRows.some((row) => row.id === selectedId)) setSelectedId(visibleRows[0].id);
  }, [visibleRows, selectedId]);

  useEffect(() => {
    if (!selected) return;
    setDraft(reviews[selected.id]?.label ?? automaticText);
  }, [selected?.id, automaticText, reviews]); // Update only when the clip or its reviewed value changes.

  const count = (id: Filter) => {
    if (id === 'all') return rows.length;
    if (id === 'reviewed') return reviewCount;
    if (id === 'unreviewed') return rows.length - reviewCount;
    return rows.filter((row) => row.disposition === id).length;
  };

  const runAuto = async () => {
    setBusy('auto'); setError(null);
    try {
      const { job } = await api.startAutoLabel({ ...auto, inputDir: cleanPath(auto.inputDir), outputDir: cleanPath(auto.outputDir), workDir: cleanPath(auto.workDir) });
      setJobs((current) => [job, ...current]);
      setValidation((current) => ({ ...current, metadataPath: job.outputs.metadataPath, audioRoot: job.outputs.audioRoot }));
      setLoad((current) => ({ ...current, metadataPath: job.outputs.metadataPath, audioRoot: job.outputs.audioRoot }));
      setNotice('Auto-label job started. Its output paths will appear below when it completes.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(null); }
  };

  const runValidation = async () => {
    setBusy('validation'); setError(null);
    try {
      const { job } = await api.startValidation({
        ...validation,
        metadataPath: cleanPath(validation.metadataPath), audioRoot: cleanPath(validation.audioRoot), outputRoot: cleanPath(validation.outputRoot),
        expectedScripts: validation.expectedScripts.split(',').map((item) => item.trim()).filter(Boolean),
        allowlist: cleanPath(validation.allowlist) || undefined,
        japaneseDictionary: cleanPath(validation.japaneseDictionary) || undefined,
        englishDictionary: cleanPath(validation.englishDictionary) || undefined,
      });
      setJobs((current) => [job, ...current]);
      setLoad((current) => ({ ...current, metadataPath: job.outputs.metadataPath, manifestPath: job.outputs.manifestPath, audioRoot: job.outputs.audioRoot }));
      setNotice('Validation job started with spellcheck evidence. It writes a new versioned run directory.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(null); }
  };

  const loadRows = async () => {
    setBusy('load'); setError(null);
    try {
      const data = await api.loadDataset({
        ...load,
        metadataPath: cleanPath(load.metadataPath), manifestPath: cleanPath(load.manifestPath) || undefined, audioRoot: cleanPath(load.audioRoot), reviewRoot: cleanPath(load.reviewRoot) || undefined, reviewer: cleanPath(load.reviewer) || undefined,
      });
      setRows(data.rows); setSession(data.session); setReviews({}); setSelectedId(data.rows[0]?.id ?? null); setSetupOpen(false);
      setNotice(`${data.rows.length.toLocaleString()} rows loaded. A fresh versioned human-review session is ready at ${data.session.directory}.`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(null); }
  };

  const save = async () => {
    if (!selected || !session) return;
    setBusy('save'); setError(null);
    const action = draft === selected.originalLabel ? 'confirmed' : 'edited';
    try {
      const { review } = await api.saveReview({ sessionId: session.id, rowId: selected.id, text: draft, action });
      setReviews((current) => ({ ...current, [selected.id]: review.human }));
      setNotice(`${action === 'edited' ? 'Human edit' : 'Confirmation'} saved with provenance. Source labels remain unchanged.`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(null); }
  };

  const selectRelative = (delta: number) => {
    const index = visibleRows.findIndex((row) => row.id === selectedId);
    const target = visibleRows[index + delta];
    if (target) setSelectedId(target.id);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">d</span><span>daiya</span><span className="brand-divider">/</span><strong>labeling</strong></div>
        <div className="topbar-actions">
          {session && <span className="session-status"><ShieldCheck size={15} aria-hidden /> {reviewCount}/{rows.length} reviewed</span>}
          <button className="button button--secondary" type="button" onClick={() => setSetupOpen((value) => !value)} aria-expanded={setupOpen}><GearSix size={17} aria-hidden /> Configure</button>
        </div>
      </header>

      {(notice || error) && <div className={`banner ${error ? 'banner--error' : ''}`} role="status"><span>{error ?? notice}</span><button type="button" aria-label="Dismiss message" onClick={() => { setNotice(null); setError(null); }}><X size={16} aria-hidden /></button></div>}

      {setupOpen && <section className="setup" aria-label="Dataset setup">
        <div className="setup-intro"><span className="eyebrow">Local workflow</span><h1>Label with context, keep the source intact.</h1><p>{configured ? 'Run each step in order. The local server prefilled its paths and options from the Whisper processor configuration, then carries every job output into the next step.' : 'Connecting to the local labeling server and loading the processor configuration…'}</p></div>
        <div className="setup-grid">
          <form className="setup-panel" onSubmit={(event) => { event.preventDefault(); void runAuto(); }}>
            <div className="panel-heading"><span className="panel-icon"><Sparkle size={18} /></span><div><h2>1. Auto-label audio</h2><p>Runs <code>auto-label</code> in the Whisper processor.</p></div></div>
            <PathField label="Input audio directory" value={auto.inputDir} onChange={(inputDir) => setAuto({ ...auto, inputDir })} placeholder="C:\datasets\raw" />
            <PathField label="New dataset output directory" value={auto.outputDir} onChange={(outputDir) => setAuto({ ...auto, outputDir })} placeholder="C:\datasets\labeled" />
            <PathField label="New pipeline work directory" value={auto.workDir} onChange={(workDir) => setAuto({ ...auto, workDir })} placeholder="C:\datasets\work" />
            <label className="check"><input type="checkbox" checked={auto.noOverlapFilter} onChange={(event) => setAuto({ ...auto, noOverlapFilter: event.target.checked })} /> Skip overlap filtering</label>
            <button className="button button--primary" disabled={busy !== null || !configured} type="submit"><Play size={16} weight="fill" />{busy === 'auto' ? 'Starting…' : 'Run auto-labeling'}</button>
          </form>

          <form className="setup-panel" onSubmit={(event) => { event.preventDefault(); void runValidation(); }}>
            <div className="panel-heading"><span className="panel-icon"><WarningCircle size={18} /></span><div><h2>2. Validate with spellcheck</h2><p>Writes spelling evidence and a candidate manifest with <code>daiya_dataset_validation</code>.</p></div></div>
            <PathField label="Pipeline metadata.jsonl" value={validation.metadataPath} onChange={(metadataPath) => setValidation({ ...validation, metadataPath })} placeholder="C:\datasets\labeled\metadata.jsonl" />
            <PathField label="Audio root" value={validation.audioRoot} onChange={(audioRoot) => setValidation({ ...validation, audioRoot })} placeholder="C:\datasets\labeled" />
            <PathField label="New validation output root" value={validation.outputRoot} onChange={(outputRoot) => setValidation({ ...validation, outputRoot })} placeholder="C:\datasets\validation-runs" />
            <div className="fields fields--two"><TextField label="Thai engine" value={validation.thaiEngine} onChange={(thaiEngine) => setValidation({ ...validation, thaiEngine })} /><TextField label="Expected scripts" value={validation.expectedScripts} onChange={(expectedScripts) => setValidation({ ...validation, expectedScripts })} /></div>
            <div className="fields fields--two"><TextField label="Review threshold" value={validation.reviewThreshold} onChange={(reviewThreshold) => setValidation({ ...validation, reviewThreshold })} /><TextField label="Minimum issues" value={validation.minIssues} onChange={(minIssues) => setValidation({ ...validation, minIssues })} /></div>
            <details><summary>Optional Japanese, English, and allowlist paths</summary><PathField label="Japanese dictionary" value={validation.japaneseDictionary} onChange={(japaneseDictionary) => setValidation({ ...validation, japaneseDictionary })} placeholder="small, core, or full" /><PathField label="English frequency dictionary" value={validation.englishDictionary} onChange={(englishDictionary) => setValidation({ ...validation, englishDictionary })} placeholder="C:\dictionaries\en.txt" /><PathField label="Versioned allowlist" value={validation.allowlist} onChange={(allowlist) => setValidation({ ...validation, allowlist })} placeholder="C:\dictionaries\terms.txt" /></details>
            <button className="button button--primary" disabled={busy !== null || !configured} type="submit"><Play size={16} weight="fill" />{busy === 'validation' ? 'Starting…' : 'Run validation'}</button>
          </form>

          <form className="setup-panel setup-panel--load" onSubmit={(event) => { event.preventDefault(); void loadRows(); }}>
            <div className="panel-heading"><span className="panel-icon"><ListBullets size={18} /></span><div><h2>3. Load the review queue</h2><p>Open any manifest; <code>keep</code> rows remain editable.</p></div></div>
            <PathField label="metadata.jsonl" value={load.metadataPath} onChange={(metadataPath) => setLoad({ ...load, metadataPath })} placeholder="C:\datasets\labeled\metadata.jsonl" />
            <PathField label="Candidate manifest.jsonl" value={load.manifestPath} onChange={(manifestPath) => setLoad({ ...load, manifestPath })} placeholder="C:\datasets\validation-runs\...\candidate-manifest.jsonl" />
            <PathField label="Audio root" value={load.audioRoot} onChange={(audioRoot) => setLoad({ ...load, audioRoot })} placeholder="C:\datasets\labeled" />
            <div className="fields fields--two"><PathField label="New empty review output directory (optional)" value={load.reviewRoot} onChange={(reviewRoot) => setLoad({ ...load, reviewRoot })} placeholder="Defaults to a new local directory" /><TextField label="Reviewer" value={load.reviewer} onChange={(reviewer) => setLoad({ ...load, reviewer })} placeholder="Defaults to Windows user" /></div>
            <button className="button button--primary" disabled={busy !== null || !configured} type="submit"><FileAudio size={16} weight="fill" />{busy === 'load' ? 'Loading…' : 'Open workbench'}</button>
          </form>
        </div>
        {jobs.length > 0 && <JobRail jobs={jobs} onRefresh={() => void refreshJobs()} onUseOutput={(outputs) => {
          setValidation((current) => ({ ...current, metadataPath: outputs.metadataPath ?? current.metadataPath, audioRoot: outputs.audioRoot ?? current.audioRoot }));
          setLoad((current) => ({ ...current, metadataPath: outputs.metadataPath ?? current.metadataPath, manifestPath: outputs.manifestPath ?? current.manifestPath, audioRoot: outputs.audioRoot ?? current.audioRoot }));
        }} />}
      </section>}

      {!rows.length && !setupOpen && <section className="empty-state"><FileAudio size={36} aria-hidden /><h2>No review queue is open</h2><p>Configure a run or load a metadata and candidate manifest pair to begin.</p><button className="button button--primary" type="button" onClick={() => setSetupOpen(true)}>Open configuration</button></section>}

      {rows.length > 0 && <section className="workbench">
        <aside className="queue" aria-label="Label queue">
          <div className="queue-head"><div><span className="eyebrow">Queue</span><strong>{rows.length.toLocaleString()} clips</strong></div><button className="icon-button" type="button" onClick={() => { setFilter('all'); setQuery(''); }} title="Clear filters"><ArrowClockwise size={17} aria-hidden /></button></div>
          <label className="search"><MagnifyingGlass size={16} aria-hidden /><span className="sr-only">Search labels</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search text, path, reason" /></label>
          <div className="filters" aria-label="Disposition filters">{FILTERS.map((item) => <button key={item.id} type="button" className={filter === item.id ? 'filter is-active' : 'filter'} onClick={() => setFilter(item.id)} aria-pressed={filter === item.id}>{item.label}<span>{count(item.id)}</span></button>)}</div>
          <div className="clip-list">{visibleRows.map((row) => <button key={row.id} type="button" className={row.id === selectedId ? 'clip is-selected' : 'clip'} onClick={() => setSelectedId(row.id)}><span className="clip-top"><span className="clip-index">{String(row.index).padStart(4, '0')}</span><Disposition value={row.disposition} /></span><span className="clip-text">{(reviews[row.id]?.label ?? row.proposedLabel ?? row.originalLabel) || '∅ Empty label'}</span><span className="clip-meta">{formatDuration(row.duration)} · {row.language}{reviews[row.id] && <Check size={14} weight="bold" aria-label="Reviewed" />}</span></button>)}</div>
        </aside>

        {selected && <article className="editor">
          <div className="editor-top"><div><span className="eyebrow">Clip {selected.index} of {rows.length}</span><h1>{selected.sourceUri || selected.id}</h1></div><div className="editor-nav"><button className="icon-button" type="button" disabled={!visibleRows.findIndex((row) => row.id === selected.id)} onClick={() => selectRelative(-1)} aria-label="Previous clip"><CaretLeft size={19} /></button><button className="icon-button" type="button" disabled={visibleRows.findIndex((row) => row.id === selected.id) === visibleRows.length - 1} onClick={() => selectRelative(1)} aria-label="Next clip"><CaretRight size={19} /></button></div></div>
          <div className="editor-layout">
            <section className="label-area">
              <div className="context-line"><Disposition value={selected.disposition} /><span>{formatDuration(selected.duration)}</span><span>{selected.language}</span>{typeof selected.sourceStart === 'number' && typeof selected.sourceEnd === 'number' && <span>source {selected.sourceStart}s–{selected.sourceEnd}s</span>}</div>
              <div className="audio-strip"><div><strong>Chunk audio</strong><span>{selected.audioAvailable ? 'Audio is served only from the loaded dataset root.' : 'No matching chunk audio was found under the audio root.'}</span></div>{selected.audioPath ? <audio controls preload="metadata" src={audioUrl(selected.audioPath)}>Your browser cannot play this audio.</audio> : <span className="audio-missing">Missing audio</span>}</div>
              <div className="source-label"><div><span className="eyebrow">Original source label</span><p>{selected.originalLabel || '∅ Empty source label'}</p></div>{selected.proposedLabel && <div><span className="eyebrow">Automatic proposal</span><p>{selected.proposedLabel}</p></div>}</div>
              <div className="editor-field"><div className="field-heading"><label htmlFor="human-label">Human label</label><span className={draft === automaticText ? 'save-state' : 'save-state is-dirty'}>{draft === automaticText ? 'No local change' : 'Unsaved change'}</span></div><textarea id="human-label" value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.altKey && event.key === 'Enter') { event.preventDefault(); void save(); } }} spellCheck="false" rows={7} /><div className="editor-actions"><button className="button button--secondary" type="button" onClick={() => setDraft(automaticText)}>Restore automatic text</button><span>Alt + Enter to save</span><button className="button button--primary" type="button" disabled={busy === 'save'} onClick={() => void save()}><FloppyDisk size={16} weight="fill" />{busy === 'save' ? 'Saving…' : reviews[selected.id] ? 'Save new revision' : 'Save human review'}</button></div></div>
            </section>
            <aside className="evidence" aria-label="Automatic evidence"><div className="evidence-heading"><span className="eyebrow">Automatic evidence</span><strong>{selected.reasons.length ? `${selected.reasons.length} review signals` : 'No review signals'}</strong></div>{selected.reasons.length > 0 && <div className="reason-list">{selected.reasons.map((reason) => <span key={reason}>{reason.replaceAll('_', ' ')}</span>)}</div>}<dl>{selected.evidence.length ? selected.evidence.map((item, index) => <div key={`${item.name}-${index}`}><dt>{item.name.replaceAll('_', ' ')}</dt><dd>{String(item.value)}{item.source && <small>{item.source}</small>}</dd></div>) : <div className="evidence-empty">This row has no attached manifest evidence.</div>}</dl><div className="provenance"><ShieldCheck size={18} aria-hidden /><p><strong>Append-only review</strong>Your decision is written to <code>{session?.directory}</code>. It never modifies source audio or labels.</p></div></aside>
          </div>
        </article>}
      </section>}
    </main>
  );
}

function TextField({ label, value, onChange, placeholder = '' }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string }) {
  return <label className="field"><span>{label}</span><input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /></label>;
}

function PathField(props: { label: string; value: string; onChange: (value: string) => void; placeholder: string }) {
  return <TextField {...props} />;
}

function JobRail({ jobs, onRefresh, onUseOutput }: { jobs: Job[]; onRefresh: () => void; onUseOutput: (outputs: Record<string, string>) => void }) {
  return <section className="job-rail"><div className="job-rail-heading"><div><span className="eyebrow">Local jobs</span><strong>Command progress</strong></div><button className="icon-button" type="button" onClick={onRefresh} aria-label="Refresh jobs"><ArrowClockwise size={17} /></button></div>{jobs.slice(0, 3).map((job) => <details key={job.id} className="job" open={job.status === 'running' || job.status === 'failed'}><summary><span><Status status={job.status} /> <strong>{job.name}</strong></span><span>{job.status === 'completed' && <button className="text-button" type="button" onClick={(event) => { event.preventDefault(); onUseOutput(job.outputs); }}>Use outputs</button>}</span></summary><div className="job-body"><code>{job.log.join('\n') || 'Waiting for command output…'}</code>{Object.entries(job.outputs).map(([key, value]) => <p key={key}><small>{key}</small><span>{value}</span></p>)}</div></details>)}</section>;
}

export default App;
