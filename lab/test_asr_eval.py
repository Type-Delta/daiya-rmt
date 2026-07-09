from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("asr_eval.py")
SPEC = importlib.util.spec_from_file_location("asr_eval", MODULE_PATH)
assert SPEC and SPEC.loader
asr_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(asr_eval)


def args(policy: str = "baseline") -> SimpleNamespace:
    return SimpleNamespace(
        beam_size=5,
        decoding_policy=policy,
        no_condition_on_previous_text=True,
        short_utterance_seconds=3.0,
        language="th",
        initial_prompt=None,
    )


def test_applied_decoder_options_use_inclusive_short_boundary() -> None:
    assert asr_eval.applied_decoder_options(args("short_beam"), 3.0) == {
        "beam_size": 8,
        "condition_on_previous_text": False,
        "temperature": [0.0, 0.8, 1.0],
        "patience": 1.2,
    }
    assert asr_eval.applied_decoder_options(args("short_beam"), 3.001)["beam_size"] == 5
    assert asr_eval.applied_decoder_options(args("short_greedy"), 2.0)["best_of"] == 1


def test_duration_bucket_boundaries() -> None:
    assert asr_eval.duration_bucket_name(1.999) == "lt_2s"
    assert asr_eval.duration_bucket_name(2.0) == "2_to_lt_3s"
    assert asr_eval.duration_bucket_name(3.0) == "3_to_lt_5s"
    assert asr_eval.duration_bucket_name(5.0) == "5_to_lt_10s"
    assert asr_eval.duration_bucket_name(10.0) == "gte_10s"
    assert asr_eval.duration_bucket_name(None) is None


def test_transcribe_forwards_and_records_expanded_options(tmp_path: Path) -> None:
    class Model:
        kwargs = None

        def transcribe(self, _path: str, **kwargs: object):
            self.kwargs = kwargs
            return [], SimpleNamespace(
                language="th",
                language_probability=1.0,
                duration=2.5,
                duration_after_vad=2.5,
            )

    model = Model()
    prediction, info = asr_eval.transcribe_one(
        model,
        tmp_path / "unused.wav",
        args("short_greedy"),
        policy_duration_seconds=2.5,
    )

    assert prediction == ""
    assert model.kwargs is not None
    assert model.kwargs["temperature"] == [0.0, 0.8, 1.0]
    assert model.kwargs["beam_size"] == 1
    assert model.kwargs["best_of"] == 1
    assert info["decoder_options"] == {
        "beam_size": 1,
        "condition_on_previous_text": False,
        "temperature": [0.0, 0.8, 1.0],
        "best_of": 1,
    }
