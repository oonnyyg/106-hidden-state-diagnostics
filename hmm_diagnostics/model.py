"""HMM model loading and validation.

Expected JSON structure::

    {
      "states": ["normal", "worn", "fault"],
      "observations": ["vibration", "heat", "noise"],
      "initial": {"normal": 0.8, "worn": 0.15, "fault": 0.05},
      "transitions": {
        "normal": {"normal": 0.8, "worn": 0.15, "fault": 0.05},
        ...
      },
      "emissions": {
        "normal": {"vibration": 0.6, "heat": 0.3, "noise": 0.1},
        ...
      },
      "sequences": [
        {"id": "line-A-1", "observations": ["vibration", "heat", null]}
      ]
    }

``null`` (or an empty string) at a time step means the observation is
missing: it carries no emission constraint while state transitions still
occur.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOLERANCE = 1e-12


class HMMValidationError(ValueError):
    """Raised when a model document fails structural or numeric checks."""


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_distribution(
    distribution: dict[str, float],
    allowed: list[str],
    label: str,
) -> None:
    if set(distribution) != set(allowed):
        unknown = set(distribution) - set(allowed)
        missing = set(allowed) - set(distribution)
        problems = []
        if unknown:
            problems.append(f"unknown keys {sorted(unknown)}")
        if missing:
            problems.append(f"missing keys {sorted(missing)}")
        raise HMMValidationError(f"{label} must define exactly {allowed}; {'; '.join(problems)}")
    for key, probability in distribution.items():
        if not _is_real_number(probability):
            raise HMMValidationError(f"{label}[{key!r}] must be a finite number")
        if not 0.0 <= float(probability) <= 1.0:
            raise HMMValidationError(f"{label}[{key!r}]={probability} is outside [0, 1]")
    total = sum(distribution.values())
    if abs(total - 1.0) > TOLERANCE:
        raise HMMValidationError(
            f"{label} probabilities must sum to 1 within {TOLERANCE:g}, got {total!r}"
        )


@dataclass(frozen=True)
class HMM:
    states: list[str]
    observations: list[str]
    initial: dict[str, float]
    transitions: dict[str, dict[str, float]]
    emissions: dict[str, dict[str, float]]
    sequences: list[dict[str, Any]]

    def emission_probability(self, state: str, symbol: str | None) -> float:
        """Return emission probability; a missing observation is unconstrained."""
        if symbol is None or symbol == "":
            return 1.0
        return self.emissions[state][symbol]


def _require_unique_names(names: list[Any], kind: str) -> list[str]:
    result: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise HMMValidationError(f"{kind} names must be non-empty strings, got {name!r}")
        if name in result:
            raise HMMValidationError(f"duplicate {kind} name: {name!r}")
        result.append(name)
    return result


def from_dict(document: dict[str, Any]) -> HMM:
    if not isinstance(document, dict):
        raise HMMValidationError("model document must be a JSON object")

    for field in ("states", "observations", "initial", "transitions", "emissions"):
        if field not in document:
            raise HMMValidationError(f"missing required field: {field!r}")

    states = _require_unique_names(document["states"], "state")
    observations = _require_unique_names(document["observations"], "observation")
    if not states:
        raise HMMValidationError("at least one state is required")
    if not observations:
        raise HMMValidationError("at least one observation symbol is required")

    initial_raw = document["initial"]
    if not isinstance(initial_raw, dict):
        raise HMMValidationError("'initial' must be an object")
    initial = {key: float(value) for key, value in initial_raw.items()}
    _validate_distribution(initial, states, "initial distribution")

    transitions_raw = document["transitions"]
    emissions_raw = document["emissions"]
    if not isinstance(transitions_raw, dict) or not isinstance(emissions_raw, dict):
        raise HMMValidationError("'transitions' and 'emissions' must be objects")

    transitions: dict[str, dict[str, float]] = {}
    emissions: dict[str, dict[str, float]] = {}
    for state in states:
        if state not in transitions_raw:
            raise HMMValidationError(f"transitions missing row for state {state!r}")
        if state not in emissions_raw:
            raise HMMValidationError(f"emissions missing row for state {state!r}")
        row_t = transitions_raw[state]
        row_e = emissions_raw[state]
        if not isinstance(row_t, dict) or not isinstance(row_e, dict):
            raise HMMValidationError(f"distribution rows for {state!r} must be objects")
        transitions[state] = {key: float(value) for key, value in row_t.items()}
        emissions[state] = {key: float(value) for key, value in row_e.items()}
        _validate_distribution(transitions[state], states, f"transitions[{state!r}]")
        _validate_distribution(emissions[state], observations, f"emissions[{state!r}]")

    sequences = document.get("sequences", [])
    if not isinstance(sequences, list):
        raise HMMValidationError("'sequences' must be a list")

    return HMM(
        states=states,
        observations=observations,
        initial=initial,
        transitions=transitions,
        emissions=emissions,
        sequences=sequences,
    )


def load_model(path: str | Path) -> HMM:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    return from_dict(document)
