"""Validate measured candidate path counters for one eager training leg."""
import json
from pathlib import Path


def verify_activation(recipe,variant,rank,steps):
    for item in recipe.get("activation", []):
        path = rank / item["file"]
        if path.parent != rank or path.suffix != ".json":
            raise ValueError(f"invalid activation receipt name: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        for key,expected in item[variant].items():
            if expected == "$steps":
                expected = steps
            actual = values[key]
            if actual != expected or type(actual) is not type(expected):
                raise ValueError(f"{path}: expected {key}={expected!r}, measured {actual!r}")
