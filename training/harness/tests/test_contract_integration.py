import hashlib
import json

from daiya_training_harness.provenance import ProvenanceRecord
from daiya_training_harness.prompting import PromptField, PromptTemplate
from daiya_training_harness.recipes import (
    BackendContract,
    BaseModel,
    ConversionSettings,
    EvaluationContract,
    FrozenManifest,
    TrainingRecipe,
)
from daiya_training_harness.splits import SplitManifest


def test_old_and_new_adapters_can_share_one_frozen_contract(tmp_path):
    manifests = {}
    split = SplitManifest(
        dataset_version="mixed-lingual-v3",
        splits={"train": ["conversation-01"], "validation": ["conversation-02"], "test": ["conversation-03"]},
    )
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(split.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for split_name in ("train", "validation", "test"):
        manifests[split_name] = FrozenManifest(
            "splits.json", hashlib.sha256(manifest_path.read_bytes()).hexdigest(), split_name
        )

    contract = TrainingRecipe(
        name="m31-2x2-v1",
        base_model=BaseModel("openai/whisper-large-v3", "refs/tags/2026-06-01"),
        manifests=tuple(manifests.values()),
        data_version="mixed-lingual-v3",
        conversion=ConversionSettings(sample_rate_hz=16_000, channels=1, sample_format="s16"),
        prompt=PromptTemplate("Context: {context}", (PromptField("context"),)),
        backend=BackendContract("ct2", "1", ("prompt_conditioning", "speaker_labels")),
        evaluation=EvaluationContract(("wer", "cer"), "wer"),
    )

    # These represent old/new harness adapters: both verify the exact same
    # frozen inputs before starting a run.
    assert contract.verify_manifests(tmp_path) == (manifest_path, manifest_path, manifest_path)
    assert contract.to_dict() == TrainingRecipe.from_dict(contract.to_dict()).to_dict()
    provenance = ProvenanceRecord(
        dataset_version=contract.data_version,
        conversion_settings=contract.conversion.__dict__,
        base_model_revision=contract.base_model.revision,
        evaluation_backend={"name": contract.backend.name, "interface_version": contract.backend.interface_version},
        split_manifest_sha256=split.sha256,
    )
    assert provenance.split_manifest_sha256 == split.sha256
