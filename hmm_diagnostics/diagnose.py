"""Batch sequence analysis and result reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hmm import ModelTables, forward_log_likelihood, viterbi
from .model import HMM

STATUS_OK = "ok"
STATUS_IMPOSSIBLE = "impossible"
STATUS_EMPTY = "empty_sequence"
STATUS_INVALID = "invalid_sequence"


@dataclass
class SequenceResult:
    sequence_id: str
    status: str
    observations: list[str | None]
    log_likelihood: float | None = None
    path: list[str] | None = None
    path_log_probability: float | None = None
    state_counts: dict[str, int] = field(default_factory=dict)
    transition_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sequence_id,
            "status": self.status,
            "observations": self.observations,
            "log_likelihood": self.log_likelihood,
            "path": self.path,
            "path_log_probability": self.path_log_probability,
            "state_counts": self.state_counts,
            "transition_counts": self.transition_counts,
            "reason": self.reason,
        }


def _normalize_sequence(raw: Any, known_observations: set[str]) -> list[str | None]:
    if not isinstance(raw, list):
        raise ValueError("observations must be a list")
    sequence: list[str | None] = []
    for symbol in raw:
        if symbol is None:
            sequence.append(None)
        elif isinstance(symbol, str):
            if symbol == "":
                sequence.append(None)
            elif symbol not in known_observations:
                raise ValueError(f"unknown observation symbol: {symbol!r}")
            else:
                sequence.append(symbol)
        else:
            raise ValueError(f"observation symbols must be strings or null, got {symbol!r}")
    return sequence


def analyze_sequence(
    model: HMM,
    sequence_id: str,
    raw_observations: Any,
    tables: ModelTables | None = None,
) -> SequenceResult:
    """Analyze one sequence; never raises for domain-level failures."""
    try:
        sequence = _normalize_sequence(raw_observations, set(model.observations))
    except ValueError as exc:
        return SequenceResult(
            sequence_id=sequence_id,
            status=STATUS_INVALID,
            observations=list(raw_observations) if isinstance(raw_observations, list) else [],
            reason=str(exc),
        )

    if not sequence:
        return SequenceResult(
            sequence_id=sequence_id,
            status=STATUS_EMPTY,
            observations=[],
            reason="empty observation sequences are not allowed",
        )

    tables = tables or ModelTables(model)
    log_likelihood = forward_log_likelihood(tables, sequence)
    best = viterbi(tables, sequence)

    if log_likelihood == float("-inf") or best is None:
        return SequenceResult(
            sequence_id=sequence_id,
            status=STATUS_IMPOSSIBLE,
            observations=sequence,
            log_likelihood=None,
            reason="sequence has zero probability under the model",
        )

    path, path_log_probability = best
    state_counts = {state: 0 for state in model.states}
    transition_counts = {
        state: {target: 0 for target in model.states} for state in model.states
    }
    for state in path:
        state_counts[state] += 1
    for source, target in zip(path, path[1:]):
        transition_counts[source][target] += 1

    return SequenceResult(
        sequence_id=sequence_id,
        status=STATUS_OK,
        observations=sequence,
        log_likelihood=log_likelihood,
        path=path,
        path_log_probability=path_log_probability,
        state_counts=state_counts,
        transition_counts=transition_counts,
    )


def analyze_all(model: HMM) -> list[SequenceResult]:
    """Analyze every sequence in input order, isolating per-sequence failures."""
    tables = ModelTables(model)
    results: list[SequenceResult] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(model.sequences):
        if not isinstance(entry, dict) or "observations" not in entry:
            results.append(
                SequenceResult(
                    sequence_id=f"sequence-{index}",
                    status=STATUS_INVALID,
                    observations=[],
                    reason="each sequence must be an object with an 'observations' list",
                )
            )
            continue
        sequence_id = entry.get("id", f"sequence-{index}")
        if not isinstance(sequence_id, str) or not sequence_id:
            sequence_id = f"sequence-{index}"
        if sequence_id in seen_ids:
            sequence_id = f"{sequence_id}#{index}"
        seen_ids.add(sequence_id)
        results.append(analyze_sequence(model, sequence_id, entry["observations"], tables))
    return results


def build_report(model: HMM, results: list[SequenceResult]) -> dict[str, Any]:
    """Assemble the full report, preserving input order."""
    summary = {
        "total": len(results),
        "ok": sum(1 for result in results if result.status == STATUS_OK),
        "impossible": sum(1 for result in results if result.status == STATUS_IMPOSSIBLE),
        "empty": sum(1 for result in results if result.status == STATUS_EMPTY),
        "invalid": sum(1 for result in results if result.status == STATUS_INVALID),
        "log_likelihoods": [
            result.log_likelihood for result in results if result.status == STATUS_OK
        ],
    }
    return {
        "states": list(model.states),
        "observations": list(model.observations),
        "results": [result.to_dict() for result in results],
        "summary": summary,
    }
