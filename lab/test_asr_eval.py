from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("asr_eval.py")
SPEC = importlib.util.spec_from_file_location("asr_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
asr_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(asr_eval)


def make_row(
    *,
    sample_id: str,
    reference: str,
    hypothesis: str,
    language: str = "en",
    mixed_bucket: str = "english",
    source_file: str = "session-a.wav",
    source_group: str = "session-a",
    duration: float = 1.0,
    latency: float = 0.25,
    model_name: str = "m2",
) -> dict:
    return {
        "status": "ok",
        "sample_id": sample_id,
        "model_name": model_name,
        "model": {"name": model_name, "path": None, "fingerprint": model_name},
        "strategy": "isolated",
        "reference": reference,
        "hypothesis": hypothesis,
        "prediction": hypothesis,
        "language_label": language,
        "mixed_bucket": mixed_bucket,
        "source_file": source_file,
        "source_group": source_group,
        "speech_duration": duration,
        "duration_seconds": duration,
        "latency_seconds": latency,
        "rtf": latency / duration,
        "english_terms": [],
        "prediction_non_thai_english_scripts": {},
        "metrics": asr_eval.text_metrics(reference, hypothesis),
        "peak_memory": {"ram_rss_bytes": 1000, "gpu_peak_bytes": 2000},
        "run": {
            "dataset_hash": "dataset",
            "manifest_hash": "manifest",
            "sample_set_hash": "samples",
            "decode_config_hash": "decode",
            "benchmark_fingerprint": "bench",
            "primary_run": True,
            "decode_config": {"initial_prompt": None},
        },
    }


def make_args(tmp_path: Path) -> Namespace:
    return Namespace(
        model=None,
        models=None,
        models_list=["m2", "m3"],
        device="cpu",
        compute_type="default",
        language=None,
        initial_prompt=None,
        beam_size=5,
        no_condition_on_previous_text=False,
        strategy="isolated",
        benchmark_strategies=None,
        requested_strategies=["isolated"],
        short_utterance_seconds=3.0,
        rolling_prompt_turns=3,
        rolling_prompt_chars=600,
        left_audio_context_seconds=4.0,
        context_max_gap_seconds=1.5,
        merge_max_seconds=12.0,
        merge_max_chunks=3,
        dataset_dir=tmp_path,
        output_dir=tmp_path,
        manifest=None,
        split_manifest=None,
        required_split="benchmark",
        selection_mode="manifest",
        bootstrap_samples=100,
        bootstrap_seed=1234,
    )


def test_manifest_ids_are_enforced_exactly_once(tmp_path: Path) -> None:
    manifest = tmp_path / "heldout.jsonl"
    manifest.write_text('{"sample_id":"b"}\n{"sample_id":"a"}\n', encoding="utf-8")

    requested = asr_eval.read_manifest_ids(manifest, "sample_id")
    metadata = [
        {"sample_id": "a", "text": "alpha", "_line_number": 1},
        {"sample_id": "b", "text": "bravo", "_line_number": 2},
    ]

    selected = asr_eval.select_metadata(metadata, requested, "sample_id")

    assert [row["_sample_id"] for row in selected] == ["b", "a"]
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        asr_eval.select_metadata(metadata, ["a", "a"], "sample_id")
    with pytest.raises(ValueError, match="missing IDs"):
        asr_eval.select_metadata(metadata, ["missing"], "sample_id")
    with pytest.raises(ValueError, match="metadata duplicates"):
        asr_eval.select_metadata(metadata + [{"sample_id": "a", "text": "again"}], ["a"], "sample_id")


def test_micro_aggregation_uses_counts_not_percentage_average() -> None:
    rows = [
        make_row(sample_id="a", reference="abcd", hypothesis="abxd"),
        make_row(sample_id="b", reference="abcdefghijklmnop", hypothesis="abcdefghijklmnop"),
    ]

    aggregate = asr_eval.aggregate_metric_rows(rows)

    assert aggregate["cer_edit_count"] == 1
    assert aggregate["cer_reference_count"] == 20
    assert aggregate["micro_cer"] == pytest.approx(0.05)
    assert aggregate["mean_cer"] == pytest.approx(0.125)


def test_summary_includes_breakdowns_and_short_utterances(tmp_path: Path) -> None:
    rows = [
        make_row(
            sample_id="a",
            reference="hello API",
            hypothesis="hello API",
            language="thai_english",
            mixed_bucket="thai_english",
            source_file="file-a.wav",
            source_group="group-a",
            duration=2.0,
        ),
        make_row(
            sample_id="b",
            reference="good morning",
            hypothesis="good mourning",
            language="en",
            mixed_bucket="english",
            source_file="file-b.wav",
            source_group="group-b",
            duration=5.0,
            model_name="m3",
        ),
        make_row(
            sample_id="a",
            reference="hello API",
            hypothesis="hello API",
            language="thai_english",
            mixed_bucket="thai_english",
            source_file="file-a.wav",
            source_group="group-a",
            duration=2.0,
            model_name="m3",
        ),
        make_row(
            sample_id="b",
            reference="good morning",
            hypothesis="good morning",
            language="en",
            mixed_bucket="english",
            source_file="file-b.wav",
            source_group="group-b",
            duration=5.0,
            model_name="m2",
        ),
    ]

    summary = asr_eval.build_summary(
        args=make_args(tmp_path),
        status="ok",
        run_id="run",
        details_path=tmp_path / "details.jsonl",
        summary_path=tmp_path / "summary.json",
        rows_seen=2,
        details=rows,
        started_at="2026-07-10T00:00:00+00:00",
        elapsed_seconds=1.0,
        run_metadata=rows[0]["run"],
    )

    assert summary["short_utterance_subset"]["count"] == 2
    assert set(summary["by_mixed_bucket"]) == {"english", "thai_english"}
    assert set(summary["by_source_group"]) == {"group-a", "group-b"}
    assert set(summary["by_model"]) == {"m2", "m3"}
    assert summary["overall"]["peak_gpu_bytes"] == 2000
    assert "m2__minus__m3" in summary["paired_model_deltas"]["isolated"]


def test_bootstrap_confidence_intervals_are_deterministic() -> None:
    rows = [
        make_row(sample_id="a", reference="abcd", hypothesis="abxd"),
        make_row(sample_id="b", reference="wxyz", hypothesis="wxyz"),
        make_row(sample_id="c", reference="hello", hypothesis="hallo"),
    ]

    first = asr_eval.bootstrap_micro_ci(rows, metric="cer", samples=200, seed=99)
    second = asr_eval.bootstrap_micro_ci(rows, metric="cer", samples=200, seed=99)

    assert first == second
    assert first["p2_5"] <= first["p50"] <= first["p97_5"]


def test_raw_comparison_rejects_incompatible_fingerprints(tmp_path: Path) -> None:
    compatible = make_row(sample_id="a", reference="abc", hypothesis="abc")
    incompatible = make_row(sample_id="a", reference="abc", hypothesis="abc")
    incompatible["run"] = {**incompatible["run"], "manifest_hash": "different"}

    with pytest.raises(ValueError, match="manifest_hash differs"):
        asr_eval.assert_compatible_raw_outputs(
            {
                tmp_path / "m2.jsonl": [compatible],
                tmp_path / "m3.jsonl": [incompatible],
            }
        )


def test_paired_bootstrap_delta_is_deterministic_and_rejects_mismatch() -> None:
    left = [
        make_row(sample_id="a", reference="abcd", hypothesis="abxd", model_name="m2"),
        make_row(sample_id="b", reference="wxyz", hypothesis="wxyz", model_name="m2"),
    ]
    right = [
        make_row(sample_id="a", reference="abcd", hypothesis="abcd", model_name="m3"),
        make_row(sample_id="b", reference="wxyz", hypothesis="wxyz", model_name="m3"),
    ]
    first = asr_eval.paired_bootstrap_delta(left, right, metric="cer", samples=200, seed=7)
    second = asr_eval.paired_bootstrap_delta(left, right, metric="cer", samples=200, seed=7)
    assert first == second
    assert first["left_minus_right"] > 0
    with pytest.raises(ValueError, match="sample sets differ"):
        asr_eval.paired_bootstrap_delta(left, right[:1], metric="cer", samples=10, seed=7)


def test_split_manifest_linkage_and_rolling_order(tmp_path: Path) -> None:
    split_manifest = tmp_path / "split.jsonl"
    split_manifest.write_text(
        '{"source_file":"call-a.wav","split":"benchmark"}\n'
        '{"source_file":"call-b.wav","split":"train"}\n',
        encoding="utf-8",
    )
    selected = [
        {
            "_sample_id": "a1",
            "source_file": r"C:\\raw\\call-a.wav",
            "source_start": 1.0,
            "file_name": "train/a1.wav",
        },
        {
            "_sample_id": "a2",
            "source_file": r"C:\\raw\\call-a.wav",
            "source_start": 2.0,
            "file_name": "train/a2.wav",
        },
    ]
    identity = asr_eval.validate_selected_split(selected, split_manifest, "benchmark")
    assert identity["validated_count"] == 2
    asr_eval.validate_rolling_order(selected)

    with pytest.raises(ValueError, match="belongs to split"):
        asr_eval.validate_selected_split(
            [{**selected[0], "_sample_id": "b1", "source_file": "call-b.wav"}],
            split_manifest,
            "benchmark",
        )
    with pytest.raises(ValueError, match="out of source-time order"):
        asr_eval.validate_rolling_order([selected[1], selected[0]])
