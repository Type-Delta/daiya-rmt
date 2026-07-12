"""Portable contracts and validation helpers for Daiya training experiments."""

from .benchmark import IncompatibleContractError, assert_comparable, run_benchmark
from .provenance import ProvenanceRecord
from .prompting import PromptField, PromptTemplate
from .recipes import TrainingRecipe
from .selection import (
    RankedCheckpoint,
    SelectionResult,
    TopKValidationProtocol,
    ValidationRecord,
    select_checkpoint,
)
from .splits import SplitManifest

__all__ = [
    "IncompatibleContractError",
    "PromptField",
    "PromptTemplate",
    "ProvenanceRecord",
    "RankedCheckpoint",
    "SelectionResult",
    "SplitManifest",
    "TopKValidationProtocol",
    "TrainingRecipe",
    "ValidationRecord",
    "assert_comparable",
    "run_benchmark",
    "select_checkpoint",
]
