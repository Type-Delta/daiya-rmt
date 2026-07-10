from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unittest

from daiya_whisper_lora.provenance import preprocessing_cache_key


@dataclass(frozen=True)
class Config:
    model_name_or_path: str = "openai/whisper-large-v3"
    language: str | None = None
    language_policy: str = "metadata"
    task: str = "transcribe"
    max_label_length: int = 448
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    output_dir: Path = Path("run")


class ProvenanceTests(unittest.TestCase):
    def test_preprocessing_cache_changes_with_split_or_prompt(self) -> None:
        common = {
            "config": Config(),
            "processor_identity": {"tokenizer": "large-v3", "sampling_rate": 16000},
        }
        base = preprocessing_cache_key(
            **common,
            dataset={"metadata_jsonl_sha256": "data", "split_manifest": {"sha256": "split-a"}},
            prompt={"enabled": True, "terms_only": True, "max_prompt_tokens": 64},
        )
        other_split = preprocessing_cache_key(
            **common,
            dataset={"metadata_jsonl_sha256": "data", "split_manifest": {"sha256": "split-b"}},
            prompt={"enabled": True, "terms_only": True, "max_prompt_tokens": 64},
        )
        other_prompt = preprocessing_cache_key(
            **common,
            dataset={"metadata_jsonl_sha256": "data", "split_manifest": {"sha256": "split-a"}},
            prompt={"enabled": False, "terms_only": True, "max_prompt_tokens": 64},
        )

        self.assertNotEqual(base, other_split)
        self.assertNotEqual(base, other_prompt)


if __name__ == "__main__":
    unittest.main()
