import hashlib
import json

import pytest

from daiya_training_harness.prompting import PromptField, PromptTemplate
from daiya_training_harness.recipes import (
    BackendContract,
    BaseModel,
    ConversionSettings,
    EvaluationContract,
    FrozenManifest,
    TrainingRecipe,
    dump_recipe,
    load_recipe,
)


@pytest.fixture
def recipe():
    return TrainingRecipe(
        name="prompt-conditioned-whisper",
        base_model=BaseModel("openai/whisper-large-v3", "refs/tags/v1.0.0"),
        manifests=(FrozenManifest("data/train.jsonl", "a" * 64),),
        data_version="thai-ja-en-2026-07",
        conversion=ConversionSettings(sample_rate_hz=16_000, channels=1),
        prompt=PromptTemplate("Context: {context}", (PromptField("context"),)),
        backend=BackendContract("legacy-or-modern", "1", ("prompt_conditioning",)),
        evaluation=EvaluationContract(("wer", "cer"), "wer"),
    )


def test_recipe_json_fixture_round_trip(tmp_path, recipe):
    path = tmp_path / "recipe.json"
    dump_recipe(recipe, path)
    assert load_recipe(path) == recipe
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 1


def test_recipe_toml_fixture_loads(tmp_path, recipe):
    path = tmp_path / "recipe.toml"
    path.write_text(
        """format_version = 1
name = "prompt-conditioned-whisper"
data_version = "thai-ja-en-2026-07"
[[manifests]]
path = "data/train.jsonl"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
split = "train"
[base_model]
model_id = "openai/whisper-large-v3"
revision = "refs/tags/v1.0.0"
[conversion]
sample_rate_hz = 16000
channels = 1
sample_format = "s16"
normalize = false
[prompt]
template = "Context: {context}"
[[prompt.fields]]
name = "context"
[backend]
name = "legacy-or-modern"
interface_version = "1"
required_features = ["prompt_conditioning"]
[evaluation]
metrics = ["wer", "cer"]
primary_metric = "wer"
greater_is_better = false
""",
        encoding="utf-8",
    )
    assert load_recipe(path) == recipe


def test_manifest_hash_verification(tmp_path, recipe):
    data = tmp_path / "data"
    data.mkdir()
    manifest = data / "train.jsonl"
    manifest.write_bytes(b"example\n")
    frozen = FrozenManifest("data/train.jsonl", hashlib.sha256(b"example\n").hexdigest())
    assert frozen.verify(tmp_path) == manifest
    with pytest.raises(ValueError, match="hash mismatch"):
        recipe.verify_manifests(tmp_path)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: BaseModel("model", ""), "revision"),
        (lambda: FrozenManifest("/absolute/data.jsonl", "a" * 64), "relative"),
        (lambda: FrozenManifest("data.jsonl", "not-a-hash"), "sha256"),
        (lambda: EvaluationContract(("wer",), "cer"), "primary"),
    ],
)
def test_invalid_contracts_are_rejected(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
