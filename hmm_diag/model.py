"""Model loading and validation for the discrete HMM."""

from __future__ import annotations

import math
from dataclasses import dataclass

PROBABILITY_TOLERANCE = 1e-12


class ModelError(ValueError):
    """Raised when the model or input document is invalid."""


def _check_probability(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{context}: probability must be a number, got {value!r}")
    prob = float(value)
    if not math.isfinite(prob):
        raise ModelError(f"{context}: probability must be finite, got {value!r}")
    if prob < 0.0 or prob > 1.0:
        raise ModelError(f"{context}: probability {prob} out of [0, 1]")
    return prob


def _check_distribution(dist: object, expected: list[str], context: str) -> dict[str, float]:
    if not isinstance(dist, dict):
        raise ModelError(f"{context}: expected an object mapping names to probabilities")
    unknown = set(dist) - set(expected)
    if unknown:
        raise ModelError(f"{context}: unknown entries {sorted(unknown)}")
    missing = set(expected) - set(dist)
    if missing:
        raise ModelError(f"{context}: missing entries {sorted(missing)}")
    probs = {name: _check_probability(dist[name], f"{context}[{name!r}]") for name in expected}
    total = math.fsum(probs.values())
    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ModelError(f"{context}: probabilities sum to {total}, not 1")
    return probs


def _check_symbols(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ModelError(f"{context}: expected a non-empty list of names")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ModelError(f"{context}: names must be non-empty strings, got {item!r}")
    if len(set(value)) != len(value):
        raise ModelError(f"{context}: names must be unique")
    return list(value)


@dataclass(frozen=True)
class HMMModel:
    """Validated discrete hidden Markov model."""

    states: list[str]
    observations: list[str]
    initial: dict[str, float]
    transition: dict[str, dict[str, float]]
    emission: dict[str, dict[str, float]]

    @classmethod
    def from_dict(cls, raw: object) -> "HMMModel":
        if not isinstance(raw, dict):
            raise ModelError("model: expected a JSON object")
        for key in ("states", "observations", "initial", "transition", "emission"):
            if key not in raw:
                raise ModelError(f"model: missing required key {key!r}")
        states = _check_symbols(raw["states"], "model.states")
        observations = _check_symbols(raw["observations"], "model.observations")
        initial = _check_distribution(raw["initial"], states, "model.initial")
        transition_raw = raw["transition"]
        if not isinstance(transition_raw, dict):
            raise ModelError("model.transition: expected an object of per-state rows")
        transition = {
            state: _check_distribution(
                transition_raw.get(state), states, f"model.transition[{state!r}]"
            )
            for state in states
        }
        emission_raw = raw["emission"]
        if not isinstance(emission_raw, dict):
            raise ModelError("model.emission: expected an object of per-state rows")
        emission = {
            state: _check_distribution(
                emission_raw.get(state), observations, f"model.emission[{state!r}]"
            )
            for state in states
        }
        return cls(states, observations, initial, transition, emission)
