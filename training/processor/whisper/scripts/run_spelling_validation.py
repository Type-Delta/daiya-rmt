"""Generate script-aware spelling evidence without modifying source labels."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Iterator

from daiya_dataset_validation.normalize import normalize_text
from daiya_dataset_validation.path_safety import reject_output_aliases
from daiya_dataset_validation.spelling import (
    PyThaiNLPChecker,
    SudachiJapaneseChecker,
    SymSpellEnglishChecker,
    validate_spelling,
)


_WorkerConfig = tuple[tuple[str, ...], str | None, str | None]
_worker_checkers: list[Any] | None = None
_worker_allowlist: frozenset[str] = frozenset()
_worker_review_threshold = 0.2
_worker_min_issues = 1
_worker_include_issue_text = False


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield row


def _allowlist(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _validate_output_path(
    output: Path,
    metadata: Path,
    allowlist: Path | None,
    english_dictionary: Path | None,
) -> None:
    reject_output_aliases(
        output,
        files=(
            ("metadata", metadata),
            ("allowlist", allowlist),
            ("English dictionary", english_dictionary),
        ),
    )


def _build_checkers(config: _WorkerConfig) -> list[Any]:
    thai_engines, japanese_dictionary, english_dictionary = config
    checkers: list[Any] = [PyThaiNLPChecker(engine) for engine in thai_engines]
    if japanese_dictionary:
        checkers.append(SudachiJapaneseChecker(japanese_dictionary))
    if english_dictionary:
        checkers.append(SymSpellEnglishChecker(english_dictionary))
    return checkers


def _checker_names(config: _WorkerConfig) -> list[str]:
    thai_engines, japanese_dictionary, english_dictionary = config
    names = [f"pythainlp-{engine}" for engine in thai_engines]
    if japanese_dictionary:
        names.append(f"sudachi-{japanese_dictionary}")
    if english_dictionary:
        names.append("symspell-en")
    return names


def _result_for_row(
    row_number: int,
    row: dict[str, Any],
    checkers: list[Any],
    allowlist: frozenset[str],
    *,
    review_threshold: float,
    min_issues: int,
    include_issue_text: bool,
) -> dict[str, Any]:
    key = str(row.get("file_name") or row.get("uri") or f"metadata.jsonl#line={row_number}").replace("\\", "/")
    label = row.get("text", row.get("label"))
    text = normalize_text(label)
    validation = validate_spelling(text, checkers, allowlist=allowlist)
    result = validation.to_dict(include_issue_text=include_issue_text)
    by_language = result["by_language"]
    result.update({
        "schema_version": "spelling-validation-1",
        "file_name": key,
        "label_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "review_recommended": (
            validation.suspicious_units >= min_issues
            and any(
                language_result["suspicious_units"] >= min_issues
                and language_result["suspicious_ratio"] >= review_threshold
                for language_result in by_language.values()
            )
        ),
    })
    return result


def _initialize_worker(
    config: _WorkerConfig,
    allowlist: frozenset[str],
    review_threshold: float,
    min_issues: int,
    include_issue_text: bool,
) -> None:
    """Initialize process-local adapters; no checker/tokenizer crosses a process boundary."""
    global _worker_checkers, _worker_allowlist, _worker_review_threshold, _worker_min_issues, _worker_include_issue_text
    _worker_checkers = _build_checkers(config)
    _worker_allowlist = allowlist
    _worker_review_threshold = review_threshold
    _worker_min_issues = min_issues
    _worker_include_issue_text = include_issue_text


def _worker_result(payload: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _worker_checkers is None:
        raise RuntimeError("spelling worker was not initialized")
    row_number, row = payload
    return _result_for_row(
        row_number,
        row,
        _worker_checkers,
        _worker_allowlist,
        review_threshold=_worker_review_threshold,
        min_issues=_worker_min_issues,
        include_issue_text=_worker_include_issue_text,
    )


def _ordered_parallel_results(
    rows: Iterable[dict[str, Any]],
    *,
    config: _WorkerConfig,
    allowlist: frozenset[str],
    review_threshold: float,
    min_issues: int,
    include_issue_text: bool,
    workers: int,
    max_in_flight: int,
    initializer: Callable[..., None] = _initialize_worker,
    worker: Callable[..., dict[str, Any]] = _worker_result,
    initargs: tuple[Any, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield process results in input order with a bounded submission window."""
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs or (config, allowlist, review_threshold, min_issues, include_issue_text),
    )
    pending: dict[int, Any] = {}
    rows_iter = iter(enumerate(rows, 1))
    next_index = 0
    next_to_emit = 0

    def fill() -> None:
        nonlocal next_index
        while len(pending) < max_in_flight:
            try:
                row_number, row = next(rows_iter)
            except StopIteration:
                return
            pending[next_index] = executor.submit(worker, (row_number, row))
            next_index += 1

    try:
        fill()
        while pending:
            done, _ = wait(tuple(pending.values()), return_when=FIRST_COMPLETED)
            for future in done:
                error = future.exception()
                if error is not None:
                    raise error
            while next_to_emit in pending and pending[next_to_emit].done():
                future = pending.pop(next_to_emit)
                yield future.result()
                next_to_emit += 1
            fill()
    except BaseException:
        for future in pending.values():
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _atomic_jsonl(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="JSONL containing file_name and text/label")
    parser.add_argument("output", type=Path)
    parser.add_argument("--thai-engine", action="append", choices=("pn", "symspellpy", "phunspell"), default=[])
    parser.add_argument("--japanese-dictionary", choices=("small", "core", "full"))
    parser.add_argument("--english-dictionary", type=Path, help="SymSpell term/count frequency dictionary")
    parser.add_argument("--allowlist", type=Path, help="UTF-8 known terms, one per line")
    parser.add_argument("--review-threshold", type=float, default=0.2)
    parser.add_argument("--min-issues", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="processes used for record validation")
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help="maximum records submitted to validation workers at once (default: workers * 2)",
    )
    parser.add_argument("--include-issue-text", action="store_true", help="Include suspicious units/suggestions in output")
    args = parser.parse_args(argv)
    if not 0 <= args.review_threshold <= 1:
        parser.error("--review-threshold must be in [0, 1]")
    if args.min_issues < 1:
        parser.error("--min-issues must be at least 1")
    if len(args.thai_engine) > 1:
        parser.error("use one --thai-engine per run so issue ratios remain comparable")

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_in_flight is not None and args.max_in_flight < 1:
        parser.error("--max-in-flight must be at least 1")
    _validate_output_path(args.output, args.metadata, args.allowlist, args.english_dictionary)
    config: _WorkerConfig = (
        tuple(args.thai_engine),
        args.japanese_dictionary,
        str(args.english_dictionary) if args.english_dictionary else None,
    )
    checker_names = _checker_names(config)
    if not checker_names:
        parser.error("enable at least one checker")
    allowlist = _allowlist(args.allowlist)

    max_in_flight = args.max_in_flight or max(1, args.workers * 2)
    if args.workers == 1:
        checkers = _build_checkers(config)
        results = (
            _result_for_row(
                row_number,
                row,
                checkers,
                allowlist,
                review_threshold=args.review_threshold,
                min_issues=args.min_issues,
                include_issue_text=args.include_issue_text,
            )
            for row_number, row in enumerate(_rows(args.metadata), 1)
        )
    else:
        results = _ordered_parallel_results(
            _rows(args.metadata),
            config=config,
            allowlist=allowlist,
            review_threshold=args.review_threshold,
            min_issues=args.min_issues,
            include_issue_text=args.include_issue_text,
            workers=args.workers,
            max_in_flight=max_in_flight,
        )
    count = 0

    def counted_results() -> Iterator[dict[str, Any]]:
        nonlocal count
        for result in results:
            count += 1
            yield result

    _atomic_jsonl(counted_results(), args.output)
    print(json.dumps({"records": count, "checkers": checker_names}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
