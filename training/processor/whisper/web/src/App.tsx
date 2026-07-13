import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
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

interface AppState {
  page: Page;
  rows: LabelRow[];
  session: Session | null;
  reviews: Record<string, ReviewState>;
  selectedId: string | null;
  draft: string;
  filter: Filter;
  query: string;
  jobs: Job[];
  busy: string | null;
  notice: string | null;
  error: string | null;
  configured: boolean;
  auto: AutoSetup;
  validation: ValidationSetup;
  load: ReviewSetup;
}

type AppAction =
  | { type: "pageChanged"; page: Page }
  | { type: "jobsRefreshed"; jobs: Job[] }
  | {
      type: "initialised";
      jobs: Job[];
      auto?: AutoSetup;
      validation?: ValidationSetup;
      load?: ReviewSetup;
      notice: string;
    }
  | { type: "autoChanged"; auto: AutoSetup }
  | { type: "validationChanged"; validation: ValidationSetup }
  | { type: "loadChanged"; load: ReviewSetup }
  | { type: "operationStarted"; operation: string }
  | { type: "operationFinished" }
  | { type: "errorSet"; error: string }
  | { type: "messageDismissed" }
  | { type: "autoStarted"; job: Job }
  | { type: "validationStarted"; job: Job }
  | { type: "jobCancelled"; job: Job; notice: string }
  | {
      type: "datasetLoaded";
      rows: LabelRow[];
      session: Session;
      reviews: Record<string, ReviewState>;
      load: ReviewSetup;
      notice: string;
    }
  | { type: "selectionChanged"; selectedId: string | null }
  | { type: "draftChanged"; draft: string }
  | { type: "filterChanged"; filter: Filter }
  | { type: "queryChanged"; query: string }
  | { type: "filtersCleared" }
  | { type: "reviewSaved"; rowId: string; review: ReviewState; notice: string }
  | { type: "jobOutputsUsed"; outputs: Job["outputs"] };

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

function initialAppState(): AppState {
  return {
    page: window.location.pathname === "/workbench" ? "workbench" : "configure",
    rows: [],
    session: null,
    reviews: {},
    selectedId: null,
    draft: "",
    filter: "all",
    query: "",
    jobs: [],
    busy: null,
    notice: null,
    error: null,
    configured: false,
    auto: INITIAL_SETUP?.auto ?? EMPTY_AUTO,
    validation: INITIAL_SETUP?.validation ?? EMPTY_VALIDATION,
    load: INITIAL_SETUP?.load ?? EMPTY_REVIEW,
  };
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "pageChanged":
      return { ...state, page: action.page };
    case "jobsRefreshed":
      return { ...state, jobs: action.jobs };
    case "initialised":
      return {
        ...state,
        auto: action.auto ?? state.auto,
        validation: action.validation ?? state.validation,
        load: action.load ?? state.load,
        jobs: action.jobs,
        configured: true,
        notice: action.notice,
      };
    case "autoChanged":
      return { ...state, auto: action.auto };
    case "validationChanged":
      return { ...state, validation: action.validation };
    case "loadChanged":
      return { ...state, load: action.load };
    case "operationStarted":
      return { ...state, busy: action.operation, error: null };
    case "operationFinished":
      return { ...state, busy: null };
    case "errorSet":
      return { ...state, error: action.error };
    case "messageDismissed":
      return { ...state, notice: null, error: null };
    case "autoStarted":
      return {
        ...state,
        jobs: [action.job, ...state.jobs],
        validation: {
          ...state.validation,
          metadataPath: action.job.outputs.metadataPath,
          audioRoot: action.job.outputs.audioRoot,
        },
        load: {
          ...state.load,
          metadataPath: action.job.outputs.metadataPath,
          audioRoot: action.job.outputs.audioRoot,
        },
        notice: "Auto-labeling started. Its output paths are ready for validation.",
      };
    case "validationStarted":
      return {
        ...state,
        jobs: [action.job, ...state.jobs],
        load: {
          ...state.load,
          metadataPath: action.job.outputs.metadataPath,
          manifestPath: action.job.outputs.manifestPath,
          audioRoot: action.job.outputs.audioRoot,
        },
        notice:
          "Validation started. Its candidate manifest is ready for the review queue.",
      };
    case "jobCancelled":
      return {
        ...state,
        jobs: state.jobs.map((job) =>
          job.id === action.job.id ? action.job : job,
        ),
        notice: action.notice,
      };
    case "datasetLoaded":
      return {
        ...state,
        page: "workbench",
        rows: action.rows,
        session: action.session,
        reviews: action.reviews,
        selectedId: action.rows[0]?.id ?? null,
        load: action.load,
        notice: action.notice,
      };
    case "selectionChanged":
      return { ...state, selectedId: action.selectedId };
    case "draftChanged":
      return { ...state, draft: action.draft };
    case "filterChanged":
      return { ...state, filter: action.filter };
    case "queryChanged":
      return { ...state, query: action.query };
    case "filtersCleared":
      return { ...state, filter: "all", query: "" };
    case "reviewSaved":
      return {
        ...state,
        reviews: { ...state.reviews, [action.rowId]: action.review },
        notice: action.notice,
      };
    case "jobOutputsUsed":
      return {
        ...state,
        validation: {
          ...state.validation,
          metadataPath:
            action.outputs.metadataPath ?? state.validation.metadataPath,
          audioRoot: action.outputs.audioRoot ?? state.validation.audioRoot,
        },
        load: {
          ...state.load,
          metadataPath: action.outputs.metadataPath ?? state.load.metadataPath,
          manifestPath: action.outputs.manifestPath ?? state.load.manifestPath,
          audioRoot: action.outputs.audioRoot ?? state.load.audioRoot,
        },
      };
  }
}

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

const ClipRow = memo(function ClipRow({
  row,
  review,
  selected,
  onSelect,
}: {
  row: LabelRow;
  review: ReviewState | undefined;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <button
      type="button"
      className={selected ? "clip is-selected" : "clip"}
      onClick={() => onSelect(row.id)}
    >
      <span className="clip-top">
        <span className="clip-index">{String(row.index).padStart(4, "0")}</span>
        <Disposition value={row.disposition} />
      </span>
      <span className="clip-text">
        {(review?.label ?? row.proposedLabel ?? row.originalLabel) || "∅ Empty label"}
      </span>
      <span className="clip-meta">
        {formatDuration(row.duration)} · {row.language}
        {review && <CheckIcon size={14} weight="bold" aria-label="Reviewed" />}
      </span>
    </button>
  );
});

function useCompactQueueLayout() {
  const [isCompact, setIsCompact] = useState(() =>
    window.matchMedia("(max-width: 780px)").matches,
  );

  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 780px)");
    const updateLayout = () => setIsCompact(mediaQuery.matches);
    updateLayout();
    mediaQuery.addEventListener("change", updateLayout);
    return () => mediaQuery.removeEventListener("change", updateLayout);
  }, []);

  return isCompact;
}

const VirtualClipList = memo(function VirtualClipList({
  rows,
  reviews,
  selectedId,
  onSelect,
}: {
  rows: LabelRow[];
  reviews: Record<string, ReviewState>;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const listRef = useRef<HTMLDivElement | null>(null);
  const isCompact = useCompactQueueLayout();
  const selectedIndex = useMemo(
    () => rows.findIndex((row) => row.id === selectedId),
    [rows, selectedId],
  );
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => listRef.current,
    estimateSize: () => (isCompact ? 242 : 100),
    horizontal: isCompact,
    overscan: 6,
  });

  useEffect(() => {
    if (selectedIndex >= 0) {
      virtualizer.scrollToIndex(selectedIndex, { align: "auto" });
    }
  }, [selectedIndex, virtualizer]);

  const virtualRows = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  return (
    <div
      ref={listRef}
      className={isCompact ? "clip-list clip-list--horizontal" : "clip-list"}
    >
      <div
        className="clip-list-inner"
        style={
          isCompact
            ? { width: `${totalSize}px`, height: "100px" }
            : { height: `${totalSize}px` }
        }
      >
        {virtualRows.map((virtualRow) => {
          const row = rows[virtualRow.index];
          return (
            <div
              key={row.id}
              ref={virtualizer.measureElement}
              data-index={virtualRow.index}
              className="clip-virtual-row"
              style={{
                transform: isCompact
                  ? `translateX(${virtualRow.start}px)`
                  : `translateY(${virtualRow.start}px)`,
              }}
            >
              <ClipRow
                row={row}
                review={reviews[row.id]}
                selected={row.id === selectedId}
                onSelect={onSelect}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
});

const ClipQueue = memo(function ClipQueue({
  filter,
  filterCounts,
  query,
  reviews,
  selectedId,
  totalRows,
  visibleRows,
  onClearFilters,
  onFilterChange,
  onQueryChange,
  onSelect,
}: {
  filter: Filter;
  filterCounts: Record<Filter, number>;
  query: string;
  reviews: Record<string, ReviewState>;
  selectedId: string | null;
  totalRows: number;
  visibleRows: LabelRow[];
  onClearFilters: () => void;
  onFilterChange: (filter: Filter) => void;
  onQueryChange: (query: string) => void;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="queue" aria-label="Label queue">
      <div className="queue-head">
        <div>
          <span className="eyebrow">Queue</span>
          <strong>{totalRows.toLocaleString()} clips</strong>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={onClearFilters}
          aria-label="Clear filters"
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
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search text, path, reason"
        />
      </label>
      <div className="filters" aria-label="Disposition filters">
        {FILTERS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={filter === item.id ? "filter is-active" : "filter"}
            onClick={() => onFilterChange(item.id)}
            aria-pressed={filter === item.id}
          >
            {item.label}
            <span>{filterCounts[item.id]}</span>
          </button>
        ))}
      </div>
      <VirtualClipList
        rows={visibleRows}
        reviews={reviews}
        selectedId={selectedId}
        onSelect={onSelect}
      />
    </aside>
  );
});

function useAppController() {
  const [state, dispatch] = useReducer(appReducer, undefined, initialAppState);
  const {
    page,
    rows,
    session,
    reviews,
    selectedId,
    draft,
    filter,
    query,
    jobs,
    busy,
    notice,
    error,
    configured,
    auto,
    validation,
    load,
  } = state;
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
    dispatch({
      type: "pageChanged",
      page: path === "/workbench" ? "workbench" : "configure",
    });
  };

  const refreshJobs = async () => {
    try {
      dispatch({ type: "jobsRefreshed", jobs: (await api.jobs()).jobs });
    } catch (cause) {
      dispatch({
        type: "errorSet",
        error: cause instanceof Error ? cause.message : String(cause),
      });
    }
  };

  useEffect(() => {
    const onPopState = () =>
      dispatch({
        type: "pageChanged",
        page:
          window.location.pathname === "/workbench"
            ? "workbench"
            : "configure",
      });
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
        dispatch({
          type: "initialised",
          jobs: jobData.jobs,
          auto: INITIAL_SETUP ? undefined : configuration.autoLabel,
          validation: INITIAL_SETUP ? undefined : configuration.validation,
          load: INITIAL_SETUP ? undefined : configuration.review,
          notice: INITIAL_SETUP
            ? "Restored saved configuration. Open the workbench to resume its review directory."
            : "Ready with project-root-relative configuration from the local server.",
        });
      } catch (cause) {
        dispatch({
          type: "errorSet",
          error: cause instanceof Error ? cause.message : String(cause),
        });
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

  const rowsById = useMemo(() => new Map(rows.map((row) => [row.id, row])), [rows]);
  const selected = selectedId ? (rowsById.get(selectedId) ?? null) : null;
  const automaticText = selected
    ? (selected.proposedLabel ?? selected.originalLabel)
    : "";
  const savedText = selected ? reviews[selected.id]?.label : undefined;
  const hasUnsavedChange = Boolean(
    selected && draft !== (savedText ?? automaticText),
  );
  const filterCounts = useMemo(() => {
    const counts: Record<Filter, number> = {
      all: rows.length,
      keep: 0,
      review: 0,
      correct: 0,
      drop: 0,
      reviewed: Object.keys(reviews).length,
      unreviewed: 0,
    };
    for (const row of rows) {
      if (
        row.disposition === "keep" ||
        row.disposition === "review" ||
        row.disposition === "correct" ||
        row.disposition === "drop"
      ) {
        counts[row.disposition] += 1;
      }
    }
    counts.unreviewed = rows.length - counts.reviewed;
    return counts;
  }, [reviews, rows]);
  const reviewCount = filterCounts.reviewed;
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
  const selectedVisibleIndex = useMemo(
    () => visibleRows.findIndex((row) => row.id === selectedId),
    [selectedId, visibleRows],
  );
  const clearFilters = useCallback(() => {
    dispatch({ type: "filtersCleared" });
  }, []);
  const changeFilter = useCallback(
    (nextFilter: Filter) =>
      dispatch({ type: "filterChanged", filter: nextFilter }),
    [],
  );
  const changeQuery = useCallback(
    (nextQuery: string) =>
      dispatch({ type: "queryChanged", query: nextQuery }),
    [],
  );
  const selectRow = useCallback(
    (nextSelectedId: string) =>
      dispatch({ type: "selectionChanged", selectedId: nextSelectedId }),
    [],
  );
  const changeAuto = useCallback(
    (nextAuto: AutoSetup) =>
      dispatch({ type: "autoChanged", auto: nextAuto }),
    [],
  );
  const changeValidation = useCallback(
    (nextValidation: ValidationSetup) =>
      dispatch({ type: "validationChanged", validation: nextValidation }),
    [],
  );
  const changeLoad = useCallback(
    (nextLoad: ReviewSetup) =>
      dispatch({ type: "loadChanged", load: nextLoad }),
    [],
  );
  const dismissMessage = useCallback(
    () => dispatch({ type: "messageDismissed" }),
    [],
  );
  const useJobOutputs = useCallback(
    (outputs: Job["outputs"]) =>
      dispatch({ type: "jobOutputsUsed", outputs }),
    [],
  );
  const changeDraft = useCallback(
    (nextDraft: string) =>
      dispatch({ type: "draftChanged", draft: nextDraft }),
    [],
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
      dispatch({ type: "selectionChanged", selectedId: visibleRows[0].id });
  }, [selectedId, visibleRows]);
  useEffect(() => {
    if (selected)
      dispatch({
        type: "draftChanged",
        draft: reviews[selected.id]?.label ?? automaticText,
      });
  }, [automaticText, reviews, selected?.id]);

  const runAuto = async () => {
    dispatch({ type: "operationStarted", operation: "auto" });
    try {
      const { job } = await api.startAutoLabel(auto);
      dispatch({ type: "autoStarted", job });
    } catch (cause) {
      dispatch({
        type: "errorSet",
        error: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      dispatch({ type: "operationFinished" });
    }
  };

  const runValidation = async () => {
    dispatch({ type: "operationStarted", operation: "validation" });
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
      dispatch({ type: "validationStarted", job });
    } catch (cause) {
      dispatch({
        type: "errorSet",
        error: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      dispatch({ type: "operationFinished" });
    }
  };

  const cancel = async (job: Job) => {
    if (
      !window.confirm(
        `Cancel ${job.name.toLowerCase()}? The current subprocess will be stopped.`,
      )
    )
      return;
    dispatch({ type: "operationStarted", operation: `cancel-${job.id}` });
    try {
      const { job: cancelled } = await api.cancelJob(job.id);
      dispatch({
        type: "jobCancelled",
        job: cancelled,
        notice: `${job.name} is being cancelled.`,
      });
    } catch (cause) {
      dispatch({
        type: "errorSet",
        error: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      dispatch({ type: "operationFinished" });
    }
  };

  const loadRows = async (request: ReviewSetup = load, restoring = false) => {
    dispatch({ type: "operationStarted", operation: "load" });
    try {
      const data = await api.loadDataset({
        ...request,
        manifestPath: request.manifestPath || undefined,
        reviewRoot: request.reviewRoot || undefined,
        reviewer: request.reviewer || undefined,
      });
      const savedReview = { ...request, reviewRoot: data.session.directory };
      localStorage.setItem(ACTIVE_REVIEW_STORAGE_KEY, JSON.stringify(savedReview));
      window.history.pushState({}, "", "/workbench");
      dispatch({
        type: "datasetLoaded",
        rows: data.rows,
        session: data.session,
        reviews: data.reviews,
        load: savedReview,
        notice: restoring || data.session.resumed
          ? `${Object.keys(data.reviews).length.toLocaleString()} saved reviews restored. You can continue where you paused.`
          : `${data.rows.length.toLocaleString()} rows loaded in a fresh review session.`,
      });
    } catch (cause) {
      dispatch({
        type: "errorSet",
        error: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      dispatch({ type: "operationFinished" });
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
    dispatch({ type: "operationStarted", operation: "save" });
    const action = draft === selected.originalLabel ? "confirmed" : "edited";
    try {
      const { review } = await api.saveReview({
        sessionId: session.id,
        rowId: selected.id,
        text: draft,
        action,
      });
      dispatch({
        type: "reviewSaved",
        rowId: selected.id,
        review: review.human,
        notice: `${action === "edited" ? "Human edit" : "Confirmation"} saved with provenance.`,
      });
    } catch (cause) {
      dispatch({
        type: "errorSet",
        error: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      dispatch({ type: "operationFinished" });
    }
  };

  const selectRelative = (delta: number) => {
    const target = visibleRows[selectedVisibleIndex + delta];
    if (target)
      dispatch({ type: "selectionChanged", selectedId: target.id });
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
        dispatch({
          type: "errorSet",
          error: cause instanceof Error ? cause.message : String(cause),
        }),
      );
  };

  const seekAndPlayAudio = (position: number) => {
    const audio = audioRef.current;
    if (!audio || !Number.isFinite(audio.duration) || audio.duration <= 0) return;
    audio.currentTime = audio.duration * position;
    void audio
      .play()
      .catch((cause) =>
        dispatch({
          type: "errorSet",
          error: cause instanceof Error ? cause.message : String(cause),
        }),
      );
  };

  const keyboardActionsRef = useRef<{
    save: () => void;
    selectRelative: (delta: number) => void;
    seekAndPlayAudio: (position: number) => void;
    startOrStopAudio: () => void;
  }>({
    save: () => {},
    selectRelative: () => {},
    seekAndPlayAudio: () => {},
    startOrStopAudio: () => {},
  });
  keyboardActionsRef.current = {
    save: () => void save(),
    selectRelative,
    seekAndPlayAudio,
    startOrStopAudio,
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
        keyboardActionsRef.current.save();
        return;
      }
      if (event.altKey && (event.key === "ArrowUp" || key === "a")) {
        event.preventDefault();
        keyboardActionsRef.current.selectRelative(-1);
        return;
      }
      if (event.altKey && (event.key === "ArrowDown" || key === "d")) {
        event.preventDefault();
        keyboardActionsRef.current.selectRelative(1);
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
        keyboardActionsRef.current.seekAndPlayAudio(Number(digitMatch[1]) / 10);
        return;
      }
      if (
        event.key === " " &&
        !editing
      ) {
        event.preventDefault();
        keyboardActionsRef.current.startOrStopAudio();
      }
    };
    document.body.addEventListener("keydown", onKeyDown, { capture: true, passive: false  });
    return () => document.body.removeEventListener("keydown", onKeyDown, { capture: true });
  }, []);

  return {
    page,
    rows,
    session,
    reviews,
    selectedId,
    draft,
    filter,
    query,
    jobs,
    busy,
    notice,
    error,
    configured,
    auto,
    validation,
    load,
    restoringWorkbench,
    selected,
    automaticText,
    hasUnsavedChange,
    filterCounts,
    reviewCount,
    visibleRows,
    selectedVisibleIndex,
    autoJob,
    validationJob,
    audioRef,
    labelRef,
    mainPanelRef,
    navigate,
    refreshJobs,
    runAuto,
    runValidation,
    cancel,
    loadRows,
    save,
    selectRelative,
    clearFilters,
    changeFilter,
    changeQuery,
    selectRow,
    changeAuto,
    changeValidation,
    changeLoad,
    dismissMessage,
    useJobOutputs,
    changeDraft,
  };
}

type AppController = ReturnType<typeof useAppController>;

function App() {
  const controller = useAppController();
  const {
    page,
    rows,
    session,
    reviews,
    selectedId,
    draft,
    filter,
    query,
    busy,
    restoringWorkbench,
    selected,
    automaticText,
    hasUnsavedChange,
    filterCounts,
    visibleRows,
    selectedVisibleIndex,
    audioRef,
    labelRef,
    mainPanelRef,
    navigate,
    save,
    selectRelative,
    clearFilters,
    changeFilter,
    changeQuery,
    selectRow,
    changeDraft,
  } = controller;

  return (
    <main className="min-h-screen">
      <AppHeader controller={controller} />

      <AppMessage controller={controller} />

      {page === "configure" && <ConfigurationPage controller={controller} />}

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
          <ClipQueue
            filter={filter}
            filterCounts={filterCounts}
            query={query}
            reviews={reviews}
            selectedId={selectedId}
            totalRows={rows.length}
            visibleRows={visibleRows}
            onClearFilters={clearFilters}
            onFilterChange={changeFilter}
            onQueryChange={changeQuery}
            onSelect={selectRow}
          />
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
                    disabled={selectedVisibleIndex <= 0}
                    onClick={() => selectRelative(-1)}
                    aria-label="Previous clip"
                  >
                    <CaretLeftIcon size={19} />
                  </button>
                  <button
                    className="icon-button"
                    type="button"
                    disabled={selectedVisibleIndex === visibleRows.length - 1}
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
                      <strong id="chunk-audio-label">Chunk audio</strong>
                      <span>
                        Ctrl + E moves focus here for Space audio. Alt + ↑/↓ or
                        Alt + A/D moves clips. 0–9 seeks and plays 0–90%.
                      </span>
                    </div>
                    {selected.audioPath ? (
                      <audio
                        ref={audioRef}
                        aria-labelledby="chunk-audio-label"
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
                      onChange={(event) => changeDraft(event.target.value)}
                      spellCheck="false"
                      rows={7}
                    />
                    <div className="editor-actions">
                      <button
                        className="button button--secondary"
                        type="button"
                        onClick={() => changeDraft(automaticText)}
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

function AppHeader({ controller }: { controller: AppController }) {
  const { page, session, reviewCount, rows, navigate } = controller;
  return (
    <header className="sticky top-0 z-10 flex min-h-[58px] items-center justify-between gap-4 border-b border-edge bg-[color-mix(in_oklch,var(--bg)_92%,var(--surface))] px-5 py-2.5 max-[780px]:px-3.5">
      <div className="flex items-center gap-2 text-sm tracking-[-0.01em]">
        <span
          className="grid h-6 w-6 place-items-center rounded-md bg-raised text-primary"
          aria-hidden
        >
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
            <ShieldCheckIcon className="text-ok" size={15} aria-hidden />{" "}
            {reviewCount}/{rows.length} reviewed
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
  );
}

function AppMessage({ controller }: { controller: AppController }) {
  const { error, notice, dismissMessage } = controller;
  if (!notice && !error) return null;
  return (
    <div
      className={`flex min-h-11 items-center justify-between gap-3 border-b px-5 py-2 text-[13px] text-ink max-[780px]:px-3.5 ${error ? "border-[color-mix(in_oklch,var(--danger)_42%,var(--edge))] bg-[color-mix(in_oklch,var(--danger)_12%,var(--bg))]" : "border-[color-mix(in_oklch,var(--primary)_32%,var(--edge))] bg-[color-mix(in_oklch,var(--primary)_10%,var(--bg))]"}`}
      role="status"
    >
      <span>{error ?? notice}</span>
      <button
        type="button"
        className="grid place-items-center border-0 bg-transparent text-inherit"
        aria-label="Dismiss message"
        onClick={dismissMessage}
      >
        <XIcon size={16} aria-hidden />
      </button>
    </div>
  );
}

function ConfigurationPage({ controller }: { controller: AppController }) {
  const { jobs, refreshJobs, useJobOutputs } = controller;
  return (
    <section className="setup" aria-label="Dataset configuration">
      <div className="setup-intro">
        <span className="eyebrow">Configuration</span>
        <h1>Prepare, validate, and open a review queue.</h1>
        <p>
          Paths and options are stored in this browser after every change. They
          are project-root-relative by default, and an existing human-review
          directory resumes its saved decisions.
        </p>
      </div>
      <div className="setup-grid">
        <AutoLabelForm controller={controller} />
        <ValidationForm controller={controller} />
        <ReviewSetupForm controller={controller} />
      </div>
      {jobs.length > 0 && (
        <JobRail
          jobs={jobs}
          onRefresh={() => void refreshJobs()}
          onUseOutput={useJobOutputs}
        />
      )}
    </section>
  );
}

function AutoLabelForm({ controller }: { controller: AppController }) {
  const { auto, autoJob, busy, configured, cancel, runAuto, changeAuto } =
    controller;
  return (
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
            Build a labeled audiofolder dataset with <code>auto-label</code>.
          </p>
        </div>
      </div>
      <PathField
        label="Input audio directory"
        value={auto.inputDir}
        onChange={(inputDir) => changeAuto({ ...auto, inputDir })}
        placeholder="training/dataset/raw"
      />
      <PathField
        label="New dataset output directory"
        value={auto.outputDir}
        onChange={(outputDir) => changeAuto({ ...auto, outputDir })}
        placeholder="training/dataset/hf_datasets/run"
        allowMissing
      />
      <PathField
        label="Pipeline work directory"
        value={auto.workDir}
        onChange={(workDir) => changeAuto({ ...auto, workDir })}
        placeholder="training/processor/whisper/work"
        allowMissing
      />
      <label className="check">
        <input
          type="checkbox"
          checked={auto.noOverlapFilter}
          onChange={(event) =>
            changeAuto({ ...auto, noOverlapFilter: event.target.checked })
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
  );
}

function ValidationForm({ controller }: { controller: AppController }) {
  const {
    validation,
    validationJob,
    busy,
    configured,
    cancel,
    runValidation,
    changeValidation,
  } = controller;
  return (
    <form
      className="setup-panel"
      onSubmit={(event) => {
        event.preventDefault();
        validationJob ? void cancel(validationJob) : void runValidation();
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
          changeValidation({ ...validation, metadataPath })
        }
        placeholder="training/dataset/hf_datasets/run/metadata.jsonl"
        kind="file"
      />
      <PathField
        label="Audio root"
        value={validation.audioRoot}
        onChange={(audioRoot) => changeValidation({ ...validation, audioRoot })}
        placeholder="training/dataset/hf_datasets/run"
      />
      <PathField
        label="Validation output root"
        value={validation.outputRoot}
        onChange={(outputRoot) =>
          changeValidation({ ...validation, outputRoot })
        }
        placeholder="training/processor/whisper/output/validation"
        allowMissing
      />
      <div className="fields fields--two">
        <TextField
          label="Thai engine"
          value={validation.thaiEngine}
          onChange={(thaiEngine) =>
            changeValidation({ ...validation, thaiEngine })
          }
        />
        <TextField
          label="Expected scripts"
          value={validation.expectedScripts}
          onChange={(expectedScripts) =>
            changeValidation({ ...validation, expectedScripts })
          }
        />
      </div>
      <div className="fields fields--two">
        <TextField
          label="Review threshold"
          value={validation.reviewThreshold}
          onChange={(reviewThreshold) =>
            changeValidation({ ...validation, reviewThreshold })
          }
        />
        <TextField
          label="Minimum issues"
          value={validation.minIssues}
          onChange={(minIssues) =>
            changeValidation({ ...validation, minIssues })
          }
        />
      </div>
      <details>
        <summary>Optional Japanese, English, and allowlist paths</summary>
        <TextField
          label="Japanese dictionary"
          value={validation.japaneseDictionary}
          onChange={(japaneseDictionary) =>
            changeValidation({ ...validation, japaneseDictionary })
          }
          placeholder="small, core, or full"
        />
        <PathField
          label="English frequency dictionary"
          value={validation.englishDictionary}
          onChange={(englishDictionary) =>
            changeValidation({ ...validation, englishDictionary })
          }
          placeholder="training/resources/en.txt"
          kind="file"
        />
        <PathField
          label="Versioned allowlist"
          value={validation.allowlist}
          onChange={(allowlist) =>
            changeValidation({ ...validation, allowlist })
          }
          placeholder="training/resources/terms.txt"
          kind="file"
        />
      </details>
      <button
        className={
          validationJob ? "button button--danger" : "button button--primary"
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
  );
}

function ReviewSetupForm({ controller }: { controller: AppController }) {
  const { load, busy, configured, loadRows, changeLoad } = controller;
  return (
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
          <p>Use a saved review directory to continue an earlier review.</p>
        </div>
      </div>
      <PathField
        label="metadata.jsonl"
        value={load.metadataPath}
        onChange={(metadataPath) => changeLoad({ ...load, metadataPath })}
        placeholder="training/dataset/hf_datasets/run/metadata.jsonl"
        kind="file"
      />
      <PathField
        label="Candidate manifest.jsonl"
        value={load.manifestPath}
        onChange={(manifestPath) => changeLoad({ ...load, manifestPath })}
        placeholder="training/processor/whisper/output/validation/.../candidate-manifest.jsonl"
        kind="file"
      />
      <PathField
        label="Audio root"
        value={load.audioRoot}
        onChange={(audioRoot) => changeLoad({ ...load, audioRoot })}
        placeholder="training/dataset/hf_datasets/run"
      />
      <div className="fields fields--two">
        <PathField
          label="Review output directory"
          value={load.reviewRoot}
          onChange={(reviewRoot) => changeLoad({ ...load, reviewRoot })}
          placeholder="training/processor/whisper/web/human-reviews/run"
          allowMissing
        />
        <TextField
          label="Reviewer"
          value={load.reviewer}
          onChange={(reviewer) => changeLoad({ ...load, reviewer })}
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
