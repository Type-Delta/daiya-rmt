from __future__ import annotations

from types import SimpleNamespace
import unittest

from daiya_whisper_lora.prompt_conditioning import (
    PromptConditioningConfig,
    build_decoder_inputs_and_labels,
    build_prompt_text,
    build_prompted_labels,
    encode_prompt_token_ids,
    mask_prompt_labels,
    parse_prompt_fields,
    validate_prompt_config,
)


class FakeTokenizer:
    bos_token_id = 1

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> SimpleNamespace:
        del add_special_tokens
        return SimpleNamespace(input_ids=[100 + index for index, _ in enumerate(text.split(), start=1)])

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token == "<|startofprev|>":
            return 50
        return None


class NoPromptTokenizer(FakeTokenizer):
    def convert_tokens_to_ids(self, token: str) -> int | None:
        del token
        return None


class PromptConditioningTests(unittest.TestCase):
    def test_build_prompt_text_extracts_terms_from_context_before_only(self) -> None:
        config = PromptConditioningConfig(enabled=True)
        prompt = build_prompt_text(
            {
                "context_before": "Topic: deploy\nTerms: Daiya, CTranslate2, QLoRA\nDo not copy this sentence.",
                "context_after": "Terms: leaked-current-label",
            },
            config,
        )

        self.assertEqual(prompt, "Terms: Daiya, CTranslate2, QLoRA")

    def test_full_context_requires_explicit_opt_in(self) -> None:
        config = PromptConditioningConfig(enabled=True, terms_only=False)
        prompt = build_prompt_text(
            {"context_before": "Topic: deploy\nTerms: Daiya"},
            config,
        )

        self.assertEqual(prompt, "Topic: deploy Terms: Daiya")

    def test_future_context_is_rejected_by_default(self) -> None:
        config = PromptConditioningConfig(enabled=True, fields=("context_after",))

        with self.assertRaisesRegex(ValueError, "future context"):
            validate_prompt_config(config)

    def test_future_context_still_honors_terms_only_when_allowed(self) -> None:
        config = PromptConditioningConfig(
            enabled=True,
            fields=("context_after",),
            allow_future_context=True,
        )
        prompt = build_prompt_text(
            {"context_after": "Topic: after current chunk\nTerms: leaked-term\nCurrent label prose."},
            config,
        )

        self.assertEqual(prompt, "Terms: leaked-term")

    def test_prompt_config_validation_is_noop_when_disabled(self) -> None:
        config = PromptConditioningConfig(enabled=False, fields=("context_after",), max_prompt_tokens=-1)

        validate_prompt_config(config)

    def test_prompt_tokens_are_bounded_and_prefixed_with_startofprev(self) -> None:
        token_ids = encode_prompt_token_ids(FakeTokenizer(), "Terms: Daiya CTranslate2 QLoRA", 2)

        self.assertEqual(token_ids, [50, 101, 102])

    def test_prompt_tokens_require_startofprev(self) -> None:
        with self.assertRaisesRegex(ValueError, "startofprev"):
            encode_prompt_token_ids(NoPromptTokenizer(), "Terms: Daiya", 2)

    def test_prompted_labels_preserve_transcript_budget_and_mask_prompt_plus_bos(self) -> None:
        prompted = build_prompted_labels(
            transcript_label_ids=[1, 10, 11, 12],
            prompt_token_ids=[50, 101, 102, 103],
            max_label_length=6,
            bos_token_id=1,
        )

        self.assertEqual(prompted.labels, [50, 101, 1, 10, 11, 12])
        self.assertEqual(prompted.prompt_label_length, 3)
        self.assertEqual(
            mask_prompt_labels(prompted.labels, prompted.prompt_label_length),
            [-100, -100, -100, 10, 11, 12],
        )

    def test_decoder_inputs_match_runtime_prompt_layout(self) -> None:
        decoder_input_ids, labels = build_decoder_inputs_and_labels(
            [50, 101, 1, 10, 11, 12],
            prompt_label_length=3,
            pad_token_id=0,
        )

        self.assertEqual(decoder_input_ids, [50, 101, 1, 10, 11, 0])
        self.assertEqual(labels, [-100, -100, 10, 11, 12, -100])

    def test_parse_prompt_fields_defaults_when_empty(self) -> None:
        self.assertEqual(parse_prompt_fields(""), ("context_before",))
        self.assertEqual(parse_prompt_fields("context_before,previous_text"), ("context_before", "previous_text"))


if __name__ == "__main__":
    unittest.main()
