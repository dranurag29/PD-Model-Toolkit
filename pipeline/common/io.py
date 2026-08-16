"""Artifact/decisions/params path conventions shared by every stage script
and by the Streamlit app, so both agree on where things live on disk."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DECISIONS_DIR = PROJECT_ROOT / "decisions"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

SAMPLES = ("development", "validation")

# Round-trip loader/dumper: reading+writing through this preserves comments,
# key order, and formatting -- critical for params.yaml, which documents
# each threshold's SAS-macro origin in comments that a plain yaml.safe_dump
# would silently discard on the first Streamlit-driven edit.
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)


def load_params() -> Dict[str, Any]:
    with open(PARAMS_PATH) as f:
        return _yaml.load(f) or {}


def save_params(params: Dict[str, Any]) -> None:
    """Writes back params.yaml. Callers should mutate the dict returned by
    load_params() in place (or a value loaded from it) rather than building
    a fresh plain dict, so ruamel's comment/formatting metadata survives."""
    with open(PARAMS_PATH, "w") as f:
        _yaml.dump(params, f)


def load_decisions(name: str) -> Dict[str, Any]:
    """Reads decisions/<name>.yaml (per-variable analyst overrides). Returns
    {'variables': {}} if the file doesn't exist yet -- callers should treat
    a missing entry as "use the stage's default_strategy"."""
    path = DECISIONS_DIR / f"{name}.yaml"
    if not path.exists():
        return {"variables": {}}
    with open(path) as f:
        data = _yaml.load(f) or {}
    data.setdefault("variables", {})
    return data


def save_decisions(name: str, data: Dict[str, Any]) -> None:
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = DECISIONS_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        _yaml.dump(data, f)


def artifact_dir(stage: str) -> Path:
    d = ARTIFACTS_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_raw(sample: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_RAW_DIR / f"{sample}_base.parquet")


def read_artifact(stage: str, sample: str) -> pd.DataFrame:
    return pd.read_parquet(artifact_dir(stage) / f"{sample}.parquet")


def write_artifact(stage: str, sample: str, df: pd.DataFrame) -> None:
    df.to_parquet(artifact_dir(stage) / f"{sample}.parquet", index=False)


def write_metrics(stage: str, metrics: Dict[str, Any]) -> None:
    with open(artifact_dir(stage) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def read_metrics(stage: str) -> Dict[str, Any]:
    path = artifact_dir(stage) / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
