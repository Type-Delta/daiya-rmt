import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowClockwiseIcon,
  CaretLeftIcon,
  CaretRightIcon,
  CheckIcon,
  FileAudioIcon,
  FloppyDiskIcon,
  FolderOpenIcon,
  GearSixIcon,
  ListBulletsIcon,
  MagnifyingGlassIcon,
  PlayIcon,
  ShieldCheckIcon,
  SparkleIcon,
  StopIcon,
  WarningCircleIcon,
  WaveformIcon,
  XIcon,
} from "@phosphor-icons/react";
import { api, audioUrl, type Job, type LabelRow, type Session } from "./api";

type Filter =
  "all" | "keep" | "review" | "correct" | "drop" | "reviewed" | "unreviewed";
type Page = "configure" | "workbench";
type AutoSetup = {
  inputDir: string;
  outputDir: string;
  workDir: string;
  noOverlapFilter: boolean;
};
type ValidationSetup = {
  metadataPath: string;
  audioRoot: string;
  outputRoot: string;
  datasetVersion: string;
  thaiEngine: string;
  expectedScripts: string;
  reviewThreshold: string;
  minIssues: string;
  allowlist: string;
  japaneseDictionary: string;
  englishDictionary: string;
};
type ReviewSetup = {
  metadataPath: string;
  manifestPath: string;
  audioRoot: string;
  reviewRoot: string;
  reviewer: string;
};
interface ReviewState {
  action: "confirmed" | "edited" | "skipped";
  label: string;
}
interface SavedSetup {
  auto: AutoSetup;
  validation: ValidationSetup;
  load: ReviewSetup;
}

const SETUP_STORAGE_KEY = "daiya-labeling-setup-v1";
const ACTIVE_REVIEW_STORAGE_KEY = "daiya-labeling-active-review-v1";
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "review", label: "Review" },
  { id: "correct", label: "Correct" },
  { id: "keep", label: "Keep" },
  { id: "drop", label: "Drop" },
  { id: "unreviewed", label: "Unreviewed" },
  { id: "reviewed", label: "Reviewed" },
];
const EMPTY_AUTO: AutoSetup = {
  inputDir: "",
  outputDir: "",
  workDir: "",
  noOverlapFilter: false,
};
const EMPTY_VALIDATION: ValidationSetup = {
  metadataPath: "",
  audioRoot: "",
  outputRoot: "",
  datasetVersion: "",
  thaiEngine: "pn",
  expectedScripts: "thai,latin",
  reviewThreshold: "0.2",
  minIssues: "1",
  allowlist: "",
  japaneseDictionary: "",
  englishDictionary: "",
};
const EMPTY_REVIEW: ReviewSetup = {
  metadataPath: "",
  manifestPath: "",
  audioRoot: "",
  reviewRoot: "",
  reviewer: "",
};

function savedSetup(): SavedSetup | null {
  try {
    const value = localStorage.getItem(SETUP_STORAGE_KEY);
    if (!value) return null;
    const parsed = JSON.parse(value) as SavedSetup;
    return parsed.auto && parsed.validation && parsed.load ? parsed : null;
  } catch {
    return null;
  }
}

function savedActiveReview(): ReviewSetup | null {
  try {
    const value = localStorage.getItem(ACTIVE_REVIEW_STORAGE_KEY);
    if (!value) return null;
    const parsed = JSON.parse(value) as ReviewSetup;
    return parsed.metadataPath && parsed.audioRoot && parsed.reviewRoot
      ? parsed
      : null;
  } catch {
    return null;
  }
}

const INITIAL_SETUP = typeof window === "undefined" ? null : savedSetup();
const INITIAL_ACTIVE_REVIEW =
  typeof window === "undefined" ? null : savedActiveReview();

function formatDuration(value: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value))
    return "Duration unknown";
  return `${value.toFixed(value < 10 ? 1 : 0)} s`;
}

function Status({ status }: { status: Job["status"] }) {
  const text =
    status === "running"
      ? "Running"
      : status === "queued"
        ? "Queued"
        : status === "completed"
          ? "Complete"
          : status === "cancelled"
            ? "Cancelled"
            : "Failed";
  return (
    <span className={`status status--${status}`}>
      <span aria-hidden />
      {text}
    </span>
  );
}

function Disposition({ value }: { value: string }) {
  return <span className={`disposition disposition--${value}`}>{value}</span>;
}

function App() {
  const [page, setPage] = useState<Page>(() =>
    window.location.pathname === "/workbench" ? "workbench" : "configure",
  );
  const [rows, setRows] = useState<LabelRow[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [reviews, setReviews] = useState<Record<string, ReviewState>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [configured, setConfigured] = useState(false);
  const [auto, setAuto] = useState<AutoSetup>(
    INITIAL_SETUP?.auto ?? EMPTY_AUTO,
  );
  const [validation, setValidation] = useState<ValidationSetup>(
    INITIAL_SETUP?.validation ?? EMPTY_VALIDATION,
  );
  const [load, setLoad] = useState<ReviewSetup>(
    INITIAL_SETUP?.load ?? EMPTY_REVIEW,
  );
  const [restoringWorkbench, setRestoringWorkbench] = useState(
    () =>
      window.location.pathname === "/workbench" && Boolean(INITIAL_ACTIVE_REVIEW),
  );
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const labelRef = useRef<HTMLTextAreaElement | null>(null);
  const mainPanelRef = useRef<HTMLElement | null>(null);
  const resumeAttemptedRef = useRef(false);

  const navigate = (path: "/" | "/workbench") => {
    window.history.pushState({}, "", path);
    setPage(path === "/workbench" ? "workbench" : "configure");
  };

  const refreshJobs = async () => {
    try {
      setJobs((await api.jobs()).jobs);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  useEffect(() => {
    const onPopState = () =>
      setPage(
        window.location.pathname === "/workbench" ? "workbench" : "configure",
      );
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    const initialise = async () => {
      try {
        const [configuration, jobData] = await Promise.all([
          api.configuration(),
          api.jobs(),
        ]);
        if (!INITIAL_SETUP) {
          setAuto(configuration.autoLabel);
          setValidation(configuration.validation);
          setLoad(configuration.review);
        }
        setJobs(jobData.jobs);
        setConfigured(true);
        setNotice(
          INITIAL_SETUP
            ? "Restored saved configuration. Open the workbench to resume its review directory."
            : "Ready with project-root-relative configuration from the local server.",
        );
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    };
    void initialise();
  }, []);

  useEffect(() => {
    if (!configured) return;
    localStorage.setItem(
      SETUP_STORAGE_KEY,
      JSON.stringify({ auto, validation, load }),
    );
  }, [auto, configured, load, validation]);

  useEffect(() => {
    if (
      !jobs.some((job) => job.status === "queued" || job.status === "running")
    )
      return;
    const id = window.setInterval(() => {
      void refreshJobs();
    }, 1000);
    return () => window.clearInterval(id);
  }, [jobs]);

  const selected = rows.find((row) => row.id === selectedId) ?? null;
  const automaticText = selected
    ? (selected.proposedLabel ?? selected.originalLabel)
    : "";
  const savedText = selected ? reviews[selected.id]?.label : undefined;
  const hasUnsavedChange = Boolean(
    selected && draft !== (savedText ?? automaticText),
  );
  const reviewCount = Object.keys(reviews).length;
  const visibleRows = useMemo(
    () =>
      rows.filter((row) => {
        const reviewed = Boolean(reviews[row.id]);
        const filterMatch =
          filter === "all" ||
          (filter === "reviewed" && reviewed) ||
          (filter === "unreviewed" && !reviewed) ||
          row.disposition === filter;
        const needle = query.trim().toLocaleLowerCase();
        return (
          filterMatch &&
          (!needle ||
            [
              row.sourceUri,
              row.originalLabel,
              row.proposedLabel ?? "",
              row.reasons.join(" "),
            ]
              .join(" ")
              .toLocaleLowerCase()
              .includes(needle))
        );
      }),
    [filter, query, reviews, rows],
  );
  const autoJob = jobs.find(
    (job) =>
      job.name === "Auto-label audio" &&
      (job.status === "queued" || job.status === "running"),
  );
  const validationJob = jobs.find(
    (job) =>
      job.name === "Validate labels and spelling" &&
      (job.status === "queued" || job.status === "running"),
  );

  useEffect(() => {
    if (visibleRows.length && !visibleRows.some((row) => row.id === selectedId))
      setSelectedId(visibleRows[0].id);
  }, [selectedId, visibleRows]);
  useEffect(() => {
    if (selected) setDraft(reviews[selected.id]?.label ?? automaticText);
  }, [automaticText, reviews, selected?.id]);

  const runAuto = async () => {
    setBusy("auto");
    setError(null);
    try {
      const { job } = await api.startAutoLabel(auto);
      setJobs((current) => [job, ...current]);
      setValidation((current) => ({
        ...current,
        metadataPath: job.outputs.metadataPath,
        audioRoot: job.outputs.audioRoot,
      }));
      setLoad((current) => ({
        ...current,
        metadataPath: job.outputs.metadataPath,
        audioRoot: job.outputs.audioRoot,
      }));
      setNotice(
        "Auto-labeling started. Its output paths are ready for validation.",
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const runValidation = async () => {
    setBusy("validation");
    setError(null);
    try {
      const { job } = await api.startValidation({
        ...validation,
        expectedScripts: validation.expectedScripts
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
        allowlist: validation.allowlist || undefined,
        japaneseDictionary: validation.japaneseDictionary || undefined,
        englishDictionary: validation.englishDictionary || undefined,
      });
      setJobs((current) => [job, ...current]);
      setLoad((current) => ({
        ...current,
        metadataPath: job.outputs.metadataPath,
        manifestPath: job.outputs.manifestPath,
        audioRoot: job.outputs.audioRoot,
      }));
      setNotice(
        "Validation started. Its candidate manifest is ready for the review queue.",
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const cancel = async (job: Job) => {
    if (
      !window.confirm(
        `Cancel ${job.name.toLowerCase()}? The current subprocess will be stopped.`,
      )
    )
      return;
    setBusy(`cancel-${job.id}`);
    setError(null);
    try {
      const { job: cancelled } = await api.cancelJob(job.id);
      setJobs((current) =>
        current.map((item) => (item.id === cancelled.id ? cancelled : item)),
      );
      setNotice(`${job.name} is being cancelled.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const loadRows = async (request: ReviewSetup = load, restoring = false) => {
    setBusy("load");
    setError(null);
    try {
      const data = await api.loadDataset({
        ...request,
        manifestPath: request.manifestPath || undefined,
        reviewRoot: request.reviewRoot || undefined,
        reviewer: request.reviewer || undefined,
      });
      setRows(data.rows);
      setSession(data.session);
      setReviews(data.reviews);
      setSelectedId(data.rows[0]?.id ?? null);
      const savedReview = { ...request, reviewRoot: data.session.directory };
      setLoad(savedReview);
      localStorage.setItem(ACTIVE_REVIEW_STORAGE_KEY, JSON.stringify(savedReview));
      setNotice(
        restoring || data.session.resumed
          ? `${Object.keys(data.reviews).length.toLocaleString()} saved reviews restored. You can continue where you paused.`
          : `${data.rows.length.toLocaleString()} rows loaded in a fresh review session.`,
      );
      navigate("/workbench");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
      if (restoring) setRestoringWorkbench(false);
    }
  };

  useEffect(() => {
    if (
      page !== "workbench" ||
      !configured ||
      rows.length > 0 ||
      !INITIAL_ACTIVE_REVIEW ||
      resumeAttemptedRef.current
    )
      return;
    resumeAttemptedRef.current = true;
    setRestoringWorkbench(true);
    void loadRows(INITIAL_ACTIVE_REVIEW, true);
  }, [configured, page, rows.length]);

  const save = async () => {
    if (!selected || !session) return;
    setBusy("save");
    setError(null);
    const action = draft === selected.originalLabel ? "confirmed" : "edited";
    try {
      const { review } = await api.saveReview({
        sessionId: session.id,
        rowId: selected.id,
        text: draft,
        action,
      });
      setReviews((current) => ({ ...current, [selected.id]: review.human }));
      setNotice(
        `${action === "edited" ? "Human edit" : "Confirmation"} saved with provenance.`,
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const selectRelative = (delta: number) => {
    const target =
      visibleRows[
        visibleRows.findIndex((row) => row.id === selectedId) + delta
      ];
    if (target) setSelectedId(target.id);
  };

  const startOrStopAudio = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (!audio.paused) {
      audio.pause();
      audio.currentTime = 0;
      return;
    }
    audio.currentTime = 0;
    void audio
      .play()
      .catch((cause) =>
        setError(cause instanceof Error ? cause.message : String(cause)),
      );
  };

  const seekAndPlayAudio = (position: number) => {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(audio.duration) || audio.duration <= 0) return;
    audio.currentTime = audio.duration * position;
    void audio
      .play()
      .catch((cause) =>
        setError(cause instanceof Error ? cause.message : String(cause)),
      );
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLocaleLowerCase();
      if (event.ctrlKey && key === "e") {
        event.preventDefault();
        if (document.activeElement === labelRef.current)
          mainPanelRef.current?.focus();
        else labelRef.current?.focus();
        return;
      }
      if (event.ctrlKey && key === "s") {
        event.preventDefault();
        void save();
        return;
      }
      if (event.altKey && (event.key === "ArrowUp" || key === "a")) {
        event.preventDefault();
        selectRelative(-1);
        return;
      }
      if (event.altKey && (event.key === "ArrowDown" || key === "d")) {
        event.preventDefault();
        selectRelative(1);
        return;
      }
      const target = event.target as HTMLElement | null;
      const editing = Boolean(
        target?.closest(
          'input, textarea, select, button, [contenteditable="true"]',
        ),
      );
      const digitMatch = event.code.match(/^(?:Digit|Numpad)([0-9])$/);
      if (!editing && digitMatch) {
        event.preventDefault();
        seekAndPlayAudio(Number(digitMatch[1]) / 10);
        return;
      }
      if (
        event.key === " " &&
        !editing
      ) {
        event.preventDefault();
        startOrStopAudio();
      }
    };
    document.body.addEventListener("keydown", onKeyDown, { capture: true, passive: false  });
    return () => document.body.removeEventListener("keydown", onKeyDown, { capture: true });
  }, [draft, reviews, selectedId, session, visibleRows]);

  const count = (id: Filter) =>
    id === "all"
      ? rows.length
      : id === "reviewed"
        ? reviewCount
        : id === "unreviewed"
          ? rows.length - reviewCount
          : rows.filter((row) => row.disposition === id).length;

  return (
    <main className="min-h-screen">
      <header className="sticky top-0 z-10 flex min-h-[58px] items-center justify-between gap-4 border-b border-edge bg-[color-mix(in_oklch,var(--bg)_92%,var(--surface))] px-5 py-2.5 max-[780px]:px-3.5">
        <div className="flex items-center gap-2 text-sm tracking-[-0.01em]">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-raised text-primary" aria-hidden>
            <WaveformIcon size={18} weight="bold" />
          </span>
          <span>daiya</span>
          <span className="text-faint">/</span>
          {page === "workbench" ? (
            <button
              className="border-0 bg-transparent p-0 font-[650] text-ink underline decoration-transparent underline-offset-[3px] transition-colors hover:text-primary hover:decoration-current"
              type="button"
              onClick={() => navigate("/")}
            >
              labeling
            </button>
          ) : (
            <strong className="font-[650]">labeling</strong>
          )}
          {page === "workbench" && (
            <>
              <span className="text-faint">/</span>
              <strong className="font-[650]">workbench</strong>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {session && page === "workbench" && (
            <span className="inline-flex items-center gap-1.5 text-xs text-muted max-[780px]:hidden">
              <ShieldCheckIcon className="text-ok" size={15} aria-hidden /> {reviewCount}/{rows.length}{" "}
              reviewed
            </span>
          )}
          {page === "workbench" && (
            <button
              className="button button--secondary"
              type="button"
              onClick={() => navigate("/")}
            >
              <GearSixIcon size={17} aria-hidden /> Configure
            </button>
          )}
        </div>
      </header>

      {(notice || error) && (
        <div className={`flex min-h-11 items-center justify-between gap-3 border-b px-5 py-2 text-[13px] text-ink max-[780px]:px-3.5 ${error ? "border-[color-mix(in_oklch,var(--danger)_42%,var(--edge))] bg-[color-mix(in_oklch,var(--danger)_12%,var(--bg))]" : "border-[color-mix(in_oklch,var(--primary)_32%,var(--edge))] bg-[color-mix(in_oklch,var(--primary)_10%,var(--bg))]"}`} role="status">
          <span>{error ?? notice}</span>
          <button
            type="button"
            className="grid place-items-center border-0 bg-transparent text-inherit"
            aria-label="Dismiss message"
            onClick={() => {
              setNotice(null);
              setError(null);
            }}
          >
            <XIcon size={16} aria-hidden />
          </button>
        </div>
      )}

      {page === "configure" && (
        <section className="setup" aria-label="Dataset configuration">
          <div className="setup-intro">
            <span className="eyebrow">Configuration</span>
            <h1>Prepare, validate, and open a review queue.</h1>
            <p>
              Paths and options are stored in this browser after every change.
              They are project-root-relative by default, and an existing
              human-review directory resumes its saved decisions.
            </p>
          </div>
          <div className="setup-grid">
            <form
              className="setup-panel"
              onSubmit={(event) => {
                event.preventDefault();
                autoJob ? void cancel(autoJob) : void runAuto();
              }}
            >
              <div className="panel-heading">
                <span className="panel-icon">
                  <SparkleIcon size={18} />
                </span>
                <div>
                  <h2>Auto-label audio</h2>
                  <p>
                    Build a labeled audiofolder dataset with{" "}
                    <code>auto-label</code>.
                  </p>
                </div>
              </div>
              <PathField
                label="Input audio directory"
                value={auto.inputDir}
                onChange={(inputDir) => setAuto({ ...auto, inputDir })}
                placeholder="training/dataset/raw"
              />
              <PathField
                label="New dataset output directory"
                value={auto.outputDir}
                onChange={(outputDir) => setAuto({ ...auto, outputDir })}
                placeholder="training/dataset/hf_datasets/run"
                allowMissing
              />
              <PathField
                label="Pipeline work directory"
                value={auto.workDir}
                onChange={(workDir) => setAuto({ ...auto, workDir })}
                placeholder="training/processor/whisper/work"
                allowMissing
              />
              <label className="check">
                <input
                  type="checkbox"
                  checked={auto.noOverlapFilter}
                  onChange={(event) =>
                    setAuto({ ...auto, noOverlapFilter: event.target.checked })
                  }
                />{" "}
                Skip overlap filtering
              </label>
              <button
                className={
                  autoJob ? "button button--danger" : "button button--primary"
                }
                disabled={!configured || busy !== null}
                type="submit"
              >
                {autoJob ? (
                  <StopIcon size={16} weight="fill" />
                ) : (
                  <PlayIcon size={16} weight="fill" />
                )}
                {autoJob
                  ? "Cancel auto-labeling"
                  : busy === "auto"
                    ? "Starting…"
                    : "Run auto-labeling"}
              </button>
              {autoJob && <JobProgress job={autoJob} />}
            </form>

            <form
              className="setup-panel"
              onSubmit={(event) => {
                event.preventDefault();
                validationJob
                  ? void cancel(validationJob)
                  : void runValidation();
              }}
            >
              <div className="panel-heading">
                <span className="panel-icon">
                  <WarningCircleIcon size={18} />
                </span>
                <div>
                  <h2>Validate with spellcheck</h2>
                  <p>Write spelling evidence and a candidate manifest.</p>
                </div>
              </div>
              <PathField
                label="Pipeline metadata.jsonl"
                value={validation.metadataPath}
                onChange={(metadataPath) =>
                  setValidation({ ...validation, metadataPath })
                }
                placeholder="training/dataset/hf_datasets/run/metadata.jsonl"
                kind="file"
              />
              <PathField
                label="Audio root"
                value={validation.audioRoot}
                onChange={(audioRoot) =>
                  setValidation({ ...validation, audioRoot })
                }
                placeholder="training/dataset/hf_datasets/run"
              />
              <PathField
                label="Validation output root"
                value={validation.outputRoot}
                onChange={(outputRoot) =>
                  setValidation({ ...validation, outputRoot })
                }
                placeholder="training/processor/whisper/output/validation"
                allowMissing
              />
              <div className="fields fields--two">
                <TextField
                  label="Thai engine"
                  value={validation.thaiEngine}
                  onChange={(thaiEngine) =>
                    setValidation({ ...validation, thaiEngine })
                  }
                />
                <TextField
                  label="Expected scripts"
                  value={validation.expectedScripts}
                  onChange={(expectedScripts) =>
                    setValidation({ ...validation, expectedScripts })
                  }
                />
              </div>
              <div className="fields fields--two">
                <TextField
                  label="Review threshold"
                  value={validation.reviewThreshold}
                  onChange={(reviewThreshold) =>
                    setValidation({ ...validation, reviewThreshold })
                  }
                />
                <TextField
                  label="Minimum issues"
                  value={validation.minIssues}
                  onChange={(minIssues) =>
                    setValidation({ ...validation, minIssues })
                  }
                />
              </div>
              <details>
                <summary>
                  Optional Japanese, English, and allowlist paths
                </summary>
                <TextField
                  label="Japanese dictionary"
                  value={validation.japaneseDictionary}
                  onChange={(japaneseDictionary) =>
                    setValidation({ ...validation, japaneseDictionary })
                  }
                  placeholder="small, core, or full"
                />
                <PathField
                  label="English frequency dictionary"
                  value={validation.englishDictionary}
                  onChange={(englishDictionary) =>
                    setValidation({ ...validation, englishDictionary })
                  }
                  placeholder="training/resources/en.txt"
                  kind="file"
                />
                <PathField
                  label="Versioned allowlist"
                  value={validation.allowlist}
                  onChange={(allowlist) =>
                    setValidation({ ...validation, allowlist })
                  }
                  placeholder="training/resources/terms.txt"
                  kind="file"
                />
              </details>
              <button
                className={
                  validationJob
                    ? "button button--danger"
                    : "button button--primary"
                }
                disabled={!configured || busy !== null}
                type="submit"
              >
                {validationJob ? (
                  <StopIcon size={16} weight="fill" />
                ) : (
                  <PlayIcon size={16} weight="fill" />
                )}
                {validationJob
                  ? "Cancel validation"
                  : busy === "validation"
                    ? "Starting…"
                    : "Run validation"}
              </button>
              {validationJob && <JobProgress job={validationJob} />}
            </form>

            <form
              className="setup-panel setup-panel--load"
              onSubmit={(event) => {
                event.preventDefault();
                void loadRows();
              }}
            >
              <div className="panel-heading">
                <span className="panel-icon">
                  <ListBulletsIcon size={18} />
                </span>
                <div>
                  <h2>Open or resume workbench</h2>
                  <p>
                    Use a saved review directory to continue an earlier review.
                  </p>
                </div>
              </div>
              <PathField
                label="metadata.jsonl"
                value={load.metadataPath}
                onChange={(metadataPath) => setLoad({ ...load, metadataPath })}
                placeholder="training/dataset/hf_datasets/run/metadata.jsonl"
                kind="file"
              />
              <PathField
                label="Candidate manifest.jsonl"
                value={load.manifestPath}
                onChange={(manifestPath) => setLoad({ ...load, manifestPath })}
                placeholder="training/processor/whisper/output/validation/.../candidate-manifest.jsonl"
                kind="file"
              />
              <PathField
                label="Audio root"
                value={load.audioRoot}
                onChange={(audioRoot) => setLoad({ ...load, audioRoot })}
                placeholder="training/dataset/hf_datasets/run"
              />
              <div className="fields fields--two">
                <PathField
                  label="Review output directory"
                  value={load.reviewRoot}
                  onChange={(reviewRoot) => setLoad({ ...load, reviewRoot })}
                  placeholder="training/processor/whisper/web/human-reviews/run"
                  allowMissing
                />
                <TextField
                  label="Reviewer"
                  value={load.reviewer}
                  onChange={(reviewer) => setLoad({ ...load, reviewer })}
                  placeholder="Defaults to Windows user"
                />
              </div>
              <button
                className="button button--primary"
                disabled={!configured || busy !== null}
                type="submit"
              >
                <FileAudioIcon size={16} weight="fill" />
                {busy === "load" ? "Opening…" : "Open workbench"}
              </button>
            </form>
          </div>
          {jobs.length > 0 && (
            <JobRail
              jobs={jobs}
              onRefresh={() => void refreshJobs()}
              onUseOutput={(outputs) => {
                setValidation((current) => ({
                  ...current,
                  metadataPath: outputs.metadataPath ?? current.metadataPath,
                  audioRoot: outputs.audioRoot ?? current.audioRoot,
                }));
                setLoad((current) => ({
                  ...current,
                  metadataPath: outputs.metadataPath ?? current.metadataPath,
                  manifestPath: outputs.manifestPath ?? current.manifestPath,
                  audioRoot: outputs.audioRoot ?? current.audioRoot,
                }));
              }}
            />
          )}
        </section>
      )}

      {page === "workbench" && !rows.length && restoringWorkbench && (
        <section className="empty-state workbench-loading" aria-live="polite">
          <ArrowClockwiseIcon className="loading-icon" size={30} aria-hidden />
          <h2>Restoring review queue</h2>
          <p>Loading saved labels, audio references, and review progress.</p>
          <div className="loading-lines" aria-hidden>
            <span />
          </div>
        </section>
      )}

      {page === "workbench" && !rows.length && !restoringWorkbench && (
        <section className="empty-state">
          <FileAudioIcon size={36} aria-hidden />
          <h2>No review queue is open</h2>
          <p>Open or resume a review directory from the configuration page.</p>
          <button
            className="button button--primary"
            type="button"
            onClick={() => navigate("/")}
          >
            Go to configuration
          </button>
        </section>
      )}

      {page === "workbench" && rows.length > 0 && (
        <section className="workbench">
          <aside className="queue" aria-label="Label queue">
            <div className="queue-head">
              <div>
                <span className="eyebrow">Queue</span>
                <strong>{rows.length.toLocaleString()} clips</strong>
              </div>
              <button
                className="icon-button"
                type="button"
                onClick={() => {
                  setFilter("all");
                  setQuery("");
                }}
                title="Clear filters"
              >
                <ArrowClockwiseIcon size={17} aria-hidden />
              </button>
            </div>
            <label className="search">
              <MagnifyingGlassIcon size={16} aria-hidden />
              <span className="sr-only">Search labels</span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search text, path, reason"
              />
            </label>
            <div className="filters" aria-label="Disposition filters">
              {FILTERS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={filter === item.id ? "filter is-active" : "filter"}
                  onClick={() => setFilter(item.id)}
                  aria-pressed={filter === item.id}
                >
                  {item.label}
                  <span>{count(item.id)}</span>
                </button>
              ))}
            </div>
            <div className="clip-list">
              {visibleRows.map((row) => (
                <button
                  key={row.id}
                  type="button"
                  className={
                    row.id === selectedId ? "clip is-selected" : "clip"
                  }
                  onClick={() => setSelectedId(row.id)}
                >
                  <span className="clip-top">
                    <span className="clip-index">
                      {String(row.index).padStart(4, "0")}
                    </span>
                    <Disposition value={row.disposition} />
                  </span>
                  <span className="clip-text">
                    {(reviews[row.id]?.label ??
                      row.proposedLabel ??
                      row.originalLabel) ||
                      "∅ Empty label"}
                  </span>
                  <span className="clip-meta">
                    {formatDuration(row.duration)} · {row.language}
                    {reviews[row.id] && (
                      <CheckIcon size={14} weight="bold" aria-label="Reviewed" />
                    )}
                  </span>
                </button>
              ))}
            </div>
          </aside>
          {selected && (
            <article className="editor">
              <div className="editor-top">
                <div>
                  <span className="eyebrow">
                    Clip {selected.index} of {rows.length}
                  </span>
                  <h1>{selected.sourceUri || selected.id}</h1>
                </div>
                <div className="editor-nav">
                  <button
                    className="icon-button"
                    type="button"
                    disabled={
                      visibleRows.findIndex((row) => row.id === selected.id) <=
                      0
                    }
                    onClick={() => selectRelative(-1)}
                    aria-label="Previous clip"
                  >
                    <CaretLeftIcon size={19} />
                  </button>
                  <button
                    className="icon-button"
                    type="button"
                    disabled={
                      visibleRows.findIndex((row) => row.id === selected.id) ===
                      visibleRows.length - 1
                    }
                    onClick={() => selectRelative(1)}
                    aria-label="Next clip"
                  >
                    <CaretRightIcon size={19} />
                  </button>
                </div>
              </div>
              <div className="editor-layout">
                <section className="label-area outline-none" ref={mainPanelRef} tabIndex={-1}>
                  <div className="context-line">
                    <Disposition value={selected.disposition} />
                    <span>{formatDuration(selected.duration)}</span>
                    <span>{selected.language}</span>
                    {typeof selected.sourceStart === "number" &&
                      typeof selected.sourceEnd === "number" && (
                        <span>
                          source {selected.sourceStart}s–{selected.sourceEnd}s
                        </span>
                      )}
                  </div>
                  <div className="audio-strip">
                    <div>
                      <strong>Chunk audio</strong>
                      <span>
                        Ctrl + E moves focus here for Space audio. Alt + ↑/↓ or
                        Alt + A/D moves clips. 0–9 seeks and plays 0–90%.
                      </span>
                    </div>
                    {selected.audioPath ? (
                      <audio
                        ref={audioRef}
                        controls
                        preload="metadata"
                        src={audioUrl(selected.audioPath)}
                      >
                        Your browser cannot play this audio.
                      </audio>
                    ) : (
                      <span className="audio-missing">Missing audio</span>
                    )}
                  </div>
                  <div className="source-label">
                    <div>
                      <span className="eyebrow">Original source label</span>
                      <p>{selected.originalLabel || "∅ Empty source label"}</p>
                    </div>
                    {selected.proposedLabel && (
                      <div>
                        <span className="eyebrow">Automatic proposal</span>
                        <p>{selected.proposedLabel}</p>
                      </div>
                    )}
                  </div>
                  <div className="editor-field">
                    <div className="field-heading">
                      <label htmlFor="human-label">Human label</label>
                      <span
                        className={
                          hasUnsavedChange
                            ? "save-state is-dirty"
                            : "save-state"
                        }
                      >
                        {hasUnsavedChange
                          ? "Unsaved change"
                          : reviews[selected.id]
                            ? "Saved revision"
                            : "Ready to confirm"}
                      </span>
                    </div>
                    <textarea
                      ref={labelRef}
                      id="human-label"
                      value={draft}
                      onChange={(event) => setDraft(event.target.value)}
                      spellCheck="false"
                      rows={7}
                    />
                    <div className="editor-actions">
                      <button
                        className="button button--secondary"
                        type="button"
                        onClick={() => setDraft(automaticText)}
                      >
                        Restore automatic text
                      </button>
                      <span>Ctrl + S saves · Ctrl + E toggles audio focus</span>
                      <button
                        className="button button--primary"
                        type="button"
                        disabled={
                          busy === "save" ||
                          (Boolean(reviews[selected.id]) && !hasUnsavedChange)
                        }
                        onClick={() => void save()}
                      >
                        <FloppyDiskIcon size={16} weight="fill" />
                        {busy === "save"
                          ? "Saving…"
                          : reviews[selected.id]
                            ? "Save revision"
                            : "Save human review"}
                      </button>
                    </div>
                  </div>
                </section>
                <aside className="evidence" aria-label="Automatic evidence">
                  <div className="evidence-heading">
                    <span className="eyebrow">Automatic evidence</span>
                    <strong>
                      {selected.reasons.length
                        ? `${selected.reasons.length} review signals`
                        : "No review signals"}
                    </strong>
                  </div>
                  {selected.reasons.length > 0 && (
                    <div className="reason-list">
                      {selected.reasons.map((reason) => (
                        <span key={reason}>{reason.replaceAll("_", " ")}</span>
                      ))}
                    </div>
                  )}
                  <dl>
                    {selected.evidence.length ? (
                      selected.evidence.map((item, index) => (
                        <div key={`${item.name}-${index}`}>
                          <dt>{item.name.replaceAll("_", " ")}</dt>
                          <dd>
                            {String(item.value)}
                            {item.source && <small>{item.source}</small>}
                          </dd>
                        </div>
                      ))
                    ) : (
                      <div className="evidence-empty">
                        This row has no attached manifest evidence.
                      </div>
                    )}
                  </dl>
                  <div className="provenance">
                    <ShieldCheckIcon size={18} aria-hidden />
                    <p>
                      <strong>
                        {session?.resumed
                          ? "Resumed review"
                          : "Append-only review"}
                      </strong>
                      Your decision is written to{" "}
                      <code>{session?.directory}</code>. It never modifies
                      source audio or labels.
                    </p>
                  </div>
                </aside>
              </div>
            </article>
          )}
        </section>
      )}
    </main>
  );
}

function TextField({
  label,
  value,
  onChange,
  placeholder = "",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
}

function PathField({
  label,
  value,
  onChange,
  placeholder,
  kind = "directory",
  allowMissing = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  kind?: "file" | "directory";
  allowMissing?: boolean;
}) {
  const [state, setState] = useState<{
    valid: boolean;
    message: string;
  } | null>(null);
  const [picking, setPicking] = useState(false);
  useEffect(() => {
    if (!value.trim()) {
      setState(null);
      return;
    }
    const timer = window.setTimeout(() => {
      void api
        .validatePath({ path: value, kind, allowMissing })
        .then((result) => setState(result))
        .catch((cause) =>
          setState({
            valid: false,
            message: cause instanceof Error ? cause.message : String(cause),
          }),
        );
    }, 300);
    return () => window.clearTimeout(timer);
  }, [allowMissing, kind, value]);
  const pick = async () => {
    setPicking(true);
    try {
      const result = await api.pickPath({ kind, initialPath: value });
      if (result.path) onChange(result.path);
    } finally {
      setPicking(false);
    }
  };
  return (
    <label className="field">
      <span>{label}</span>
      <span className="field-control">
        <input
          className={state && !state.valid ? "is-invalid" : ""}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          aria-invalid={state ? !state.valid : undefined}
        />
        <button
          className="picker-button"
          type="button"
          onClick={() => void pick()}
          disabled={picking}
          aria-label={`Choose ${label}`}
        >
          <FolderOpenIcon size={16} aria-hidden />
        </button>
      </span>
      {state && (
        <small className={state.valid ? "path-state" : "path-state is-invalid"}>
          {state.message}
        </small>
      )}
    </label>
  );
}

function JobProgress({ job }: { job: Job }) {
  const percent = Math.round(job.progress.fraction * 100);
  return (
    <div
      className="job-progress"
      aria-label={`${job.name}: ${job.progress.detail}`}
    >
      <div className="job-progress-track">
        <span
          className={
            job.status === "running" && percent === 0 ? "is-indeterminate" : ""
          }
          style={{
            width: `${Math.max(job.status === "running" ? 4 : 0, percent)}%`,
          }}
        />
      </div>
      <small>
        {job.progress.detail} · {percent}%
      </small>
    </div>
  );
}

function JobRail({
  jobs,
  onRefresh,
  onUseOutput,
}: {
  jobs: Job[];
  onRefresh: () => void;
  onUseOutput: (outputs: Record<string, string>) => void;
}) {
  return (
    <section className="job-rail">
      <div className="job-rail-heading">
        <div>
          <span className="eyebrow">Local jobs</span>
          <strong>Command progress</strong>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={onRefresh}
          aria-label="Refresh jobs"
        >
          <ArrowClockwiseIcon size={17} />
        </button>
      </div>
      {jobs.slice(0, 3).map((job) => (
        <details
          key={job.id}
          className="job"
          open={job.status === "running" || job.status === "failed"}
        >
          <summary>
            <span>
              <Status status={job.status} /> <strong>{job.name}</strong>
            </span>
            <span>
              {job.status === "completed" && (
                <button
                  className="text-button"
                  type="button"
                  onClick={(event) => {
                    event.preventDefault();
                    onUseOutput(job.outputs);
                  }}
                >
                  Use outputs
                </button>
              )}
            </span>
          </summary>
          <div className="job-body">
            <JobProgress job={job} />
            <code>{job.log.join("\n") || "Waiting for command output…"}</code>
            {Object.entries(job.outputs).map(([key, value]) => (
              <p key={key}>
                <small>{key}</small>
                <span>{value}</span>
              </p>
            ))}
          </div>
        </details>
      ))}
    </section>
  );
}

export default App;
