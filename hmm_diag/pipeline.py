"""Batch analysis of observation sequences against a validated model."""

from __future__ import annotations

from collections import Counter

from .algorithms import forward_log_likelihood, is_missing, viterbi
from .model import HMMModel, ModelError


def _failure(sequence_id: object, reason: str) -> dict:
    return {
        "id": sequence_id,
        "status": "failed",
        "reason": reason,
        "log_likelihood": None,
        "path": None,
        "timeline": None,
        "state_counts": None,
        "transition_counts": None,
    }


def analyze_sequence(model: HMMModel, entry: object) -> dict:
    if not isinstance(entry, dict):
        return _failure(None, "sequence entry must be an object with 'id' and 'observations'")
    sequence_id = entry.get("id")
    if not isinstance(sequence_id, str) or not sequence_id:
        return _failure(sequence_id, "sequence id must be a non-empty string")
    observations = entry.get("observations")
    if not isinstance(observations, list):
        return _failure(sequence_id, "observations must be a list")
    if not observations:
        return _failure(sequence_id, "observation sequence is empty")
    for observation in observations:
        if is_missing(observation):
            continue
        if not isinstance(observation, str) or observation not in model.observations:
            return _failure(sequence_id, f"unknown observation symbol {observation!r}")
    return _analyze_valid(model, sequence_id, observations)


def _analyze_valid(model: HMMModel, sequence_id: str, observations: list) -> dict:
    log_likelihood = forward_log_likelihood(model, observations)
    decoded = viterbi(model, observations)
    if decoded is None or log_likelihood == float("-inf"):
        result = _failure(sequence_id, "sequence is impossible under the model (zero probability)")
        result["status"] = "impossible"
        return result

    path, path_log_prob = decoded
    timeline = [
        {
            "time": index,
            "observation": None if is_missing(obs) else obs,
            "state": state,
        }
        for index, (obs, state) in enumerate(zip(observations, path))
    ]
    state_counts = {state: 0 for state in model.states}
    state_counts.update(Counter(path))
    transition_counts: dict[str, dict[str, int]] = {
        src: {dst: 0 for dst in model.states} for src in model.states
    }
    for src, dst in zip(path, path[1:]):
        transition_counts[src][dst] += 1
    return {
        "id": sequence_id,
        "status": "ok",
        "reason": None,
        "log_likelihood": log_likelihood,
        "path_log_probability": path_log_prob,
        "path": path,
        "timeline": timeline,
        "state_counts": state_counts,
        "transition_counts": transition_counts,
    }


def analyze_document(raw: object) -> dict:
    """Analyze a full input document, preserving sequence order."""
    if not isinstance(raw, dict) or "model" not in raw or "sequences" not in raw:
        raise ModelError("input document must contain 'model' and 'sequences'")
    model = HMMModel.from_dict(raw["model"])
    sequences = raw["sequences"]
    if not isinstance(sequences, list):
        raise ModelError("sequences must be a list")
    results = [analyze_sequence(model, entry) for entry in sequences]
    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "impossible": sum(1 for r in results if r["status"] == "impossible"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
    }
    return {"summary": summary, "results": results}
