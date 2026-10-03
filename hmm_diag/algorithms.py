"""Forward log-likelihood and Viterbi decoding in log space."""

from __future__ import annotations

import math

from .model import HMMModel

NEG_INF = float("-inf")


def is_missing(observation: object) -> bool:
    """A null or empty-string observation imposes no emission constraint."""
    return observation is None or observation == ""


def _log(value: float) -> float:
    return math.log(value) if value > 0.0 else NEG_INF


def _log_sum_exp(values: list[float]) -> float:
    peak = max(values)
    if peak == NEG_INF:
        return NEG_INF
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


def _emission_log_probs(model: HMMModel, observation: object) -> dict[str, float]:
    if is_missing(observation):
        return {state: 0.0 for state in model.states}
    if observation not in model.observations:
        raise KeyError(observation)
    return {state: _log(model.emission[state][observation]) for state in model.states}


def forward_log_likelihood(model: HMMModel, observations: list[object]) -> float:
    """Log-likelihood of the full observation sequence via the forward algorithm."""
    if not observations:
        raise ValueError("observation sequence must not be empty")
    emit = _emission_log_probs(model, observations[0])
    alpha = {s: _log(model.initial[s]) + emit[s] for s in model.states}
    for observation in observations[1:]:
        emit = _emission_log_probs(model, observation)
        alpha = {
            s: _log_sum_exp(
                [alpha[prev] + _log(model.transition[prev][s]) for prev in model.states]
            )
            + emit[s]
            for s in model.states
        }
    return _log_sum_exp(list(alpha.values()))


def viterbi(model: HMMModel, observations: list[object]) -> tuple[list[str], float] | None:
    """Most likely joint state path; ties broken by lexicographic state-name sequence.

    Returns None when no state path has positive probability.
    """
    if not observations:
        raise ValueError("observation sequence must not be empty")
    emit = _emission_log_probs(model, observations[0])
    # Each candidate carries its full state-name tuple so ties resolve lexicographically.
    best: dict[str, tuple[float, tuple[str, ...]]] = {}
    for state in model.states:
        score = _log(model.initial[state]) + emit[state]
        if score > NEG_INF:
            best[state] = (score, (state,))
    for observation in observations[1:]:
        emit = _emission_log_probs(model, observation)
        nxt: dict[str, tuple[float, tuple[str, ...]]] = {}
        for state in model.states:
            if emit[state] == NEG_INF:
                continue
            candidates = []
            for prev, (score, path) in best.items():
                trans = _log(model.transition[prev][state])
                if trans > NEG_INF:
                    candidates.append((score + trans, path))
            if not candidates:
                continue
            top_score = max(score for score, _ in candidates)
            top_path = min(path for score, path in candidates if score == top_score)
            nxt[state] = (top_score + emit[state], top_path + (state,))
        best = nxt
        if not best:
            return None
    final_score = max(score for score, _ in best.values())
    final_path = min(path for score, path in best.values() if score == final_score)
    return list(final_path), final_score
