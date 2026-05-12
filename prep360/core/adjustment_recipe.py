"""Serializable adjustment recipes for the shared color pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


RECIPE_VERSION = 1


@dataclass
class LUTSettings:
    enabled: bool = False
    path: str = ""
    strength: float = 1.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            strength = float(self.strength)
        except (TypeError, ValueError):
            errors.append("LUT strength must be a number")
        else:
            if not 0.0 <= strength <= 1.0:
                errors.append("LUT strength must be between 0.0 and 1.0")

        if self.enabled:
            if not str(self.path).strip():
                errors.append("LUT is enabled but no LUT path is set")
            elif not Path(self.path).exists():
                errors.append(f"LUT file not found: {self.path}")
        return errors


@dataclass
class ToneSettings:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0


@dataclass
class WhiteBalanceSettings:
    temperature: float = 0.0
    tint: float = 0.0


@dataclass
class ColorSettings:
    saturation: float = 0.0
    vibrance: float = 0.0


@dataclass
class DetailSettings:
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.0
    sharpen_threshold: float = 0.0
    denoise_strength: float = 0.0
    denoise_method: str = "bilateral"


@dataclass
class CorrectionSettings:
    clahe_clip: float = 0.0
    vignette_strength: float = 0.0


@dataclass
class OutputSettings:
    format: str = "jpg"
    quality: int = 99
    bit_depth: str = "8-bit"
    copy_exif: bool = True
    embed_icc: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        fmt = str(self.format).lower()
        if fmt not in {"jpg", "jpeg", "png", "tiff", "tif"}:
            errors.append(f"Unsupported output format: {self.format}")
        try:
            quality = int(self.quality)
        except (TypeError, ValueError):
            errors.append("Output quality must be an integer")
        else:
            if not 1 <= quality <= 100:
                errors.append("Output quality must be between 1 and 100")
        if self.bit_depth not in {"8-bit", "16-bit"}:
            errors.append("Output bit depth must be '8-bit' or '16-bit'")
        return errors


@dataclass
class AdjustmentRecipe:
    version: int = RECIPE_VERSION
    name: str = "Untitled Adjustment"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    input_lut: LUTSettings = field(default_factory=LUTSettings)
    tone: ToneSettings = field(default_factory=ToneSettings)
    white_balance: WhiteBalanceSettings = field(default_factory=WhiteBalanceSettings)
    color: ColorSettings = field(default_factory=ColorSettings)
    detail: DetailSettings = field(default_factory=DetailSettings)
    corrections: CorrectionSettings = field(default_factory=CorrectionSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdjustmentRecipe":
        if not isinstance(data, Mapping):
            raise ValueError("Adjustment recipe JSON must be an object")

        version = int(data.get("version", RECIPE_VERSION))
        if version > RECIPE_VERSION:
            raise ValueError(f"Unsupported adjustment recipe version: {version}")

        pipeline = data.get("color_pipeline", {})
        if not isinstance(pipeline, Mapping):
            raise ValueError("Adjustment recipe color_pipeline must be an object")

        def section(name: str) -> Mapping[str, Any]:
            value = pipeline.get(name, data.get(name, {}))
            return value if isinstance(value, Mapping) else {}

        return cls(
            version=version,
            name=str(data.get("name", "Untitled Adjustment")),
            created_at=str(data.get("created_at", datetime.now().isoformat(timespec="seconds"))),
            input_lut=_dataclass_from_mapping(LUTSettings, section("input_lut")),
            tone=_dataclass_from_mapping(ToneSettings, section("tone")),
            white_balance=_dataclass_from_mapping(WhiteBalanceSettings, section("white_balance")),
            color=_dataclass_from_mapping(ColorSettings, section("color")),
            detail=_dataclass_from_mapping(DetailSettings, section("detail")),
            corrections=_dataclass_from_mapping(CorrectionSettings, section("corrections")),
            output=_dataclass_from_mapping(OutputSettings, data.get("output", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
            "color_pipeline": {
                "input_lut": asdict(self.input_lut),
                "tone": asdict(self.tone),
                "white_balance": asdict(self.white_balance),
                "color": asdict(self.color),
                "detail": asdict(self.detail),
                "corrections": asdict(self.corrections),
            },
            "output": asdict(self.output),
        }

    @classmethod
    def load(cls, path: str | Path) -> "AdjustmentRecipe":
        recipe_path = Path(path)
        try:
            with open(recipe_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Adjustment recipe not found: {recipe_path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid adjustment recipe JSON: {recipe_path}: {e}") from e
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        recipe_path = Path(path)
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.version != RECIPE_VERSION:
            errors.append(f"Unsupported adjustment recipe version: {self.version}")
        if not str(self.name).strip():
            errors.append("Recipe name cannot be empty")
        errors.extend(self.input_lut.validate())
        errors.extend(_validate_numbers("tone", self.tone))
        errors.extend(_validate_numbers("white_balance", self.white_balance))
        errors.extend(_validate_numbers("color", self.color))
        errors.extend(_validate_numbers("detail", self.detail, skip={"denoise_method"}))
        errors.extend(_validate_numbers("corrections", self.corrections))
        errors.extend(self.output.validate())
        return errors


def _dataclass_from_mapping(cls, data: Any):
    data = data if isinstance(data, Mapping) else {}
    allowed = set(cls.__dataclass_fields__.keys())
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _validate_numbers(section_name: str, obj: object, skip: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    skip = skip or set()
    for key, value in asdict(obj).items():
        if key in skip:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{section_name}.{key} must be a number")
            continue
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            errors.append(f"{section_name}.{key} must be finite")
    return errors


__all__ = [
    "RECIPE_VERSION",
    "LUTSettings",
    "ToneSettings",
    "WhiteBalanceSettings",
    "ColorSettings",
    "DetailSettings",
    "CorrectionSettings",
    "OutputSettings",
    "AdjustmentRecipe",
]
