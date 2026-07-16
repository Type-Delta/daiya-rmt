"""Portable, frozen configuration for prompt-conditioned training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tomllib
from typing import Any, Mapping

from .prompting import PromptTemplate


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class BaseModel:
    model_id: str
    revision: str

    def __post_init__(self) -> None:
        _require_text("base model id", self.model_id)
        _require_text("base model revision", self.revision)


@dataclass(frozen=True)
class FrozenManifest:
    path: str
    sha256: str
    split: str = "train"

    def __post_init__(self) -> None:
        _require_text("manifest path", self.path)
        _require_text("manifest split", self.split)
        path = Path(self.path)
        if (
            path.is_absolute()
            or PurePosixPath(self.path).is_absolute()
            or PureWindowsPath(self.path).is_absolute()
            or ".." in path.parts
            or ".." in PurePosixPath(self.path).parts
            or ".." in PureWindowsPath(self.path).parts
        ):
            raise ValueError("manifest paths must be portable relative paths")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("manifest sha256 must contain exactly 64 hexadecimal characters")

    def verify(self, root: str | Path = ".") -> Path:
        target = Path(root) / self.path
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != self.sha256.lower():
            raise ValueError(f"manifest hash mismatch for {self.path!r}")
        return target


@dataclass(frozen=True)
class ConversionSettings:
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_format: str = "s16"
    normalize: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("conversion sample rate must be positive")
        if self.channels <= 0:
            raise ValueError("conversion channel count must be positive")
        _require_text("conversion sample format", self.sample_format)


@dataclass(frozen=True)
class BackendContract:
    name: str
    interface_version: str
    required_features: tuple[str, ...] = ("prompt_conditioning",)

    def __post_init__(self) -> None:
        _require_text("backend name", self.name)
        _require_text("backend interface version", self.interface_version)
        if not self.required_features or any(not item for item in self.required_features):
            raise ValueError("backend required_features must be non-empty strings")


@dataclass(frozen=True)
class EvaluationContract:
    metrics: tuple[str, ...]
    primary_metric: str
    greater_is_better: bool = False

    def __post_init__(self) -> None:
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("evaluation metrics must be a non-empty unique list")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary evaluation metric must occur in metrics")


@dataclass(frozen=True)
class TrainingRecipe:
    """Versioned contract consumable by both legacy and current harnesses."""

    name: str
    base_model: BaseModel
    manifests: tuple[FrozenManifest, ...]
    data_version: str
    conversion: ConversionSettings
    prompt: PromptTemplate
    backend: BackendContract
    evaluation: EvaluationContract
    format_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        _require_text("recipe name", self.name)
        _require_text("data version", self.data_version)
        if not self.manifests:
            raise ValueError("recipe must include at least one frozen manifest")
        splits = [manifest.split for manifest in self.manifests]
        if len(splits) != len(set(splits)):
            raise ValueError("manifest splits must be unique")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["prompt"] = self.prompt.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRecipe":
        version = value.get("format_version", 1)
        if version != 1:
            raise ValueError(f"unsupported recipe format_version: {version!r}")
        return cls(
            name=str(value["name"]),
            base_model=BaseModel(**value["base_model"]),
            manifests=tuple(FrozenManifest(**item) for item in value["manifests"]),
            data_version=str(value["data_version"]),
            conversion=ConversionSettings(**value["conversion"]),
            prompt=PromptTemplate.from_dict(value["prompt"]),
            backend=BackendContract(
                **{**value["backend"], "required_features": tuple(value["backend"].get("required_features", ("prompt_conditioning",)))}
            ),
            evaluation=EvaluationContract(
                **{**value["evaluation"], "metrics": tuple(value["evaluation"]["metrics"])}
            ),
        )

    def verify_manifests(self, root: str | Path = ".") -> tuple[Path, ...]:
        return tuple(manifest.verify(root) for manifest in self.manifests)


def load_recipe(path: str | Path) -> TrainingRecipe:
    source = Path(path)
    with source.open("rb") as stream:
        if source.suffix.lower() == ".toml":
            value = tomllib.load(stream)
        elif source.suffix.lower() == ".json":
            value = json.load(stream)
        else:
            raise ValueError("recipe files must use .json or .toml")
    if not isinstance(value, Mapping):
        raise TypeError("recipe root must be an object/table")
    return TrainingRecipe.from_dict(value)


def dump_recipe(recipe: TrainingRecipe, path: str | Path) -> None:
    """Write canonical JSON; TOML is intentionally read-only in the stdlib."""

    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("recipes can only be written as .json")
    target.write_text(
        json.dumps(recipe.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
