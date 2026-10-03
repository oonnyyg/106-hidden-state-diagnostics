"""Forward and Viterbi inference in log space.

All probabilities are converted to log probabilities up front so that long
observation sequences never underflow.  Probability zero maps to negative
infinity and represents an impossible event.
"""

from __future__ import annotations

import math

NEG_INF = float("-inf")


def _safe_log(value: float) -> float:
    if value <= 0.0:
        return NEG_INF
    return math.log(value)


def _logaddexp(values: list[float]) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


class ModelTables:
    """Precomputed log-probability tables keyed by state index."""

    def __init__(self, model) -> None:
        self.states = list(model.states)
        self.state_index = {state: i for i, state in enumerate(self.states)}
        self.observations = set(model.observations)
        self.log_initial = [_safe_log(model.initial[s]) for s in self.states]
        self.log_transition = [
            [_safe_log(model.transitions[s][t]) for t in self.states]
            for s in self.states
        ]
        self.log_emission = {
            s: {symbol: _safe_log(model.emissions[s][symbol]) for symbol in model.observations}
            for s in self.states
        }

    def emit_log(self, state_index: int, symbol: str | None) -> float:
        if symbol is None or symbol == "":
            return 0.0
        return self.log_emission[self.states[state_index]][symbol]


def forward_log_likelihood(tables: ModelTables, sequence: list[str | None]) -> float:
    """Return log P(sequence) using the forward algorithm."""
    alpha = [
        tables.log_initial[i] + tables.emit_log(i, sequence[0])
        for i in range(len(tables.states))
    ]
    for t in range(1, len(sequence)):
        symbol = sequence[t]
        new_alpha: list[float] = []
        for j in range(len(tables.states)):
            incoming = [alpha[i] + tables.log_transition[i][j] for i in range(len(tables.states))]
            new_alpha.append(_logaddexp(incoming) + tables.emit_log(j, symbol))
        alpha = new_alpha
    return _logaddexp(alpha)


def viterbi(tables: ModelTables, sequence: list[str | None]) -> tuple[list[str], float] | None:
    """Return (state names, joint log probability) of the best full path.

    Ties are broken by the lexicographic order of the complete state-name
    sequence: states are iterated in lexicographic order and a strict ``>``
    comparison keeps the lexicographically smallest predecessor / final
    state.  Returns ``None`` when no path has positive probability.
    """
    order = sorted(range(len(tables.states)), key=lambda i: tables.states[i])
    count = len(order)

    delta = [
        tables.log_initial[i] + tables.emit_log(i, sequence[0])
        for i in range(count)
    ]
    backpointers: list[list[int]] = [[-1] * count]

    for t in range(1, len(sequence)):
        symbol = sequence[t]
        new_delta: list[float] = [NEG_INF] * count
        pointer: list[int] = [-1] * count
        for j in order:
            best_score = NEG_INF
            best_predecessor = -1
            for i in order:
                score = delta[i] + tables.log_transition[i][j]
                if score > best_score:
                    best_score = score
                    best_predecessor = i
            if best_predecessor != -1:
                new_delta[j] = best_score + tables.emit_log(j, symbol)
                pointer[j] = best_predecessor
        delta = new_delta
        backpointers.append(pointer)

    best_final = -1
    best_score = NEG_INF
    for j in order:
        if delta[j] > best_score:
            best_score = delta[j]
            best_final = j
    if best_final == -1 or best_score == NEG_INF:
        return None

    path_indices = [best_final]
    for t in range(len(sequence) - 1, 0, -1):
        path_indices.append(backpointers[t][path_indices[-1]])
    path_indices.reverse()
    return [tables.states[i] for i in path_indices], best_score
