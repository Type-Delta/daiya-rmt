"""Prompt templates with an explicit, portable input contract."""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptField:
    """A value accepted by a :class:`PromptTemplate`."""

    name: str
    required: bool = True
    default: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid prompt field name: {self.name!r}")
        if self.required and self.default is not None:
            raise ValueError(f"required field {self.name!r} cannot have a default")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptField":
        return cls(
            name=str(value["name"]),
            required=bool(value.get("required", True)),
            default=value.get("default"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "required": self.required}
        if self.default is not None:
            result["default"] = self.default
        return result


@dataclass(frozen=True)
class PromptTemplate:
    """A format-string template whose inputs are declared and validated."""

    template: str
    fields: tuple[PromptField, ...]

    def __post_init__(self) -> None:
        if not self.template:
            raise ValueError("prompt template cannot be empty")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("prompt field names must be unique")

        referenced: set[str] = set()
        try:
            parsed = Formatter().parse(self.template)
            for _, name, format_spec, conversion in parsed:
                if name is None:
                    continue
                if not name.isidentifier():
                    raise ValueError(
                        "prompt placeholders must be simple identifiers, "
                        f"not {name!r}"
                    )
                if format_spec or conversion:
                    raise ValueError("prompt placeholders cannot use formatting or conversion")
                referenced.add(name)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid prompt template: {exc}") from exc

        declared = set(names)
        if referenced != declared:
            missing = sorted(declared - referenced)
            unknown = sorted(referenced - declared)
            details = []
            if missing:
                details.append(f"unused fields: {', '.join(missing)}")
            if unknown:
                details.append(f"undeclared placeholders: {', '.join(unknown)}")
            raise ValueError("prompt field contract mismatch (" + "; ".join(details) + ")")

    def render(self, values: Mapping[str, object], *, strict: bool = True) -> str:
        declared = {field.name: field for field in self.fields}
        if strict:
            unexpected = sorted(set(values) - set(declared))
            if unexpected:
                raise ValueError(f"unexpected prompt fields: {', '.join(unexpected)}")

        resolved: dict[str, object] = {}
        missing: list[str] = []
        for name, field in declared.items():
            if name in values:
                resolved[name] = values[name]
            elif field.default is not None:
                resolved[name] = field.default
            elif field.required:
                missing.append(name)
            else:
                resolved[name] = ""
        if missing:
            raise ValueError(f"missing required prompt fields: {', '.join(sorted(missing))}")
        return self.template.format_map(resolved)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptTemplate":
        raw_fields = value.get("fields", ())
        if not isinstance(raw_fields, (list, tuple)):
            raise TypeError("prompt fields must be a list")
        return cls(
            template=str(value["template"]),
            fields=tuple(PromptField.from_dict(field) for field in raw_fields),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "fields": [field.to_dict() for field in self.fields],
        }
