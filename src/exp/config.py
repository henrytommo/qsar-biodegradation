"""config/best_params.yml -- source of truth for each model's tuned params, scaling method,
and selected features. `tune` prints/--writes a block; drivers read it via load(), falling
back to the model's in-module default when an entry is missing."""
import datetime
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "best_params.yml"


def load(name):
    """(params, method, features) for `name` from the YAML. Raises if there is no entry --
    the config is the single source of truth, so a model must be tuned before it can be run.
    features is None when unset -> use all features."""
    data = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    entry = (data or {}).get(name)
    if not entry or "params" not in entry:
        raise SystemExit(f"No tuned params for '{name}' in {CONFIG_PATH.name} -- "
                         f"run: python -m exp.tune {name} --write")
    return entry["params"], entry.get("method", "standardize"), entry.get("features")


def build_entry(name, best, features, metrics):
    """One model's config entry (dict), with provenance. Casts away numpy types so the YAML
    stays plain scalars."""
    return {name: {
        "params": {k: _plain(v) for k, v in best["params"].items()},
        "method": best["method"],
        "n_features": best.get("n_features") or (len(features) if features else None),
        "features": list(features) if features else None,
        "provenance": {
            "tuned": datetime.date.today().isoformat(),
            "cv_f1": round(float(best["f1_mean"]), 3),
            "test_f1": round(float(metrics["f1"]), 3),
        },
    }}


def to_yaml(entry):
    return yaml.safe_dump(entry, sort_keys=False, default_flow_style=False)


def write(entry):
    """Merge `entry` into config/best_params.yml, leaving other models untouched."""
    data = {}
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    data.update(entry)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _plain(v):
    """numpy/py scalar -> plain python scalar for clean YAML."""
    if hasattr(v, "item"):
        return v.item()
    return v
