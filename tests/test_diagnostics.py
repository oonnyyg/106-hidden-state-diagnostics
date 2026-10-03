import itertools
import math

import pytest

from hmm_diagnostics.diagnose import (
    STATUS_EMPTY,
    STATUS_IMPOSSIBLE,
    STATUS_INVALID,
    STATUS_OK,
    analyze_all,
    analyze_sequence,
    build_report,
)
from hmm_diagnostics.hmm import ModelTables, forward_log_likelihood, viterbi
from hmm_diagnostics.model import HMMValidationError, from_dict, load_model


def make_doc(initial=None, transitions=None, emissions=None, sequences=None):
    states = ["s1", "s2"]
    observations = ["a", "b"]
    return {
        "states": states,
        "observations": observations,
        "initial": initial or {"s1": 0.5, "s2": 0.5},
        "transitions": transitions
        or {
            "s1": {"s1": 0.5, "s2": 0.5},
            "s2": {"s1": 0.5, "s2": 0.5},
        },
        "emissions": emissions
        or {
            "s1": {"a": 0.8, "b": 0.2},
            "s2": {"a": 0.3, "b": 0.7},
        },
        "sequences": sequences or [],
    }


def test_validation_rejects_bad_sums_and_ranges():
    doc = make_doc(initial={"s1": 0.5, "s2": 0.4})
    with pytest.raises(HMMValidationError):
        from_dict(doc)
    doc = make_doc(initial={"s1": 1.2, "s2": -0.2})
    with pytest.raises(HMMValidationError):
        from_dict(doc)


def test_validation_rejects_duplicate_names():
    doc = make_doc()
    doc["states"] = ["s1", "s1"]
    with pytest.raises(HMMValidationError):
        from_dict(doc)


def test_validation_accepts_sum_within_tolerance():
    doc = make_doc(initial={"s1": 0.5, "s2": 0.5 + 5e-13})
    from_dict(doc)


def test_empty_sequence_is_rejected():
    model = from_dict(make_doc(sequences=[{"id": "e", "observations": []}]))
    results = analyze_all(model)
    assert results[0].status == STATUS_EMPTY
    assert results[0].path is None


def test_unknown_symbol_is_invalid_not_impossible():
    model = from_dict(make_doc())
    result = analyze_sequence(model, "x", ["a", "zzz"])
    assert result.status == STATUS_INVALID
    assert "unknown observation" in result.reason


def test_missing_observation_is_unconstrained_and_still_transitions():
    transitions = {
        "s1": {"s1": 0.1, "s2": 0.9},
        "s2": {"s1": 0.2, "s2": 0.8},
    }
    model = from_dict(make_doc(transitions=transitions))
    tables = ModelTables(model)
    with_missing = ["a", None, "b"]
    with_marker = ["a", "", "b"]
    assert forward_log_likelihood(tables, with_missing) == pytest.approx(
        forward_log_likelihood(tables, with_marker)
    )
    result = analyze_sequence(model, "m", with_missing)
    assert result.status == STATUS_OK
    assert len(result.path) == 3
    total_transitions = sum(sum(row.values()) for row in result.transition_counts.values())
    assert total_transitions == 2


def test_impossible_zero_initial_probability():
    doc = make_doc(
        initial={"s1": 1.0, "s2": 0.0},
        emissions={
            "s1": {"a": 1.0, "b": 0.0},
            "s2": {"a": 0.0, "b": 1.0},
        },
    )
    model = from_dict(doc)
    assert analyze_sequence(model, "ok", ["a"]).status == STATUS_OK
    result = analyze_sequence(model, "bad", ["b"])
    assert result.status == STATUS_IMPOSSIBLE
    assert result.log_likelihood is None
    assert result.path is None


def test_impossible_zero_transition_does_not_block_other_sequences():
    doc = make_doc()
    # s1 only emits "a" and must move to s2; s2 only emits "b" and stays.
    doc["emissions"] = {
        "s1": {"a": 1.0, "b": 0.0},
        "s2": {"a": 0.0, "b": 1.0},
    }
    doc["initial"] = {"s1": 1.0, "s2": 0.0}
    doc["transitions"] = {
        "s1": {"s1": 0.0, "s2": 1.0},
        "s2": {"s1": 0.0, "s2": 1.0},
    }
    doc["sequences"] = [
        {"id": "good", "observations": ["a", "b"]},
        {"id": "bad", "observations": ["b"]},
        {"id": "also-good", "observations": ["a"]},
    ]
    model = from_dict(doc)
    results = analyze_all(model)
    assert [r.sequence_id for r in results] == ["good", "bad", "also-good"]
    assert [r.status for r in results] == [STATUS_OK, STATUS_IMPOSSIBLE, STATUS_OK]


def test_viterbi_finds_global_optimum_not_greedy_path():
    # Greedy picks s2 at t=0 (initial 0.6 > 0.4).  s2's self-loop is leaky,
    # so over 40 steps starting in the stable s1 wins globally.
    doc = {
        "states": ["s1", "s2"],
        "observations": ["b"],
        "initial": {"s1": 0.4, "s2": 0.6},
        "transitions": {
            "s1": {"s1": 1.0, "s2": 0.0},
            "s2": {"s1": 0.1, "s2": 0.9},
        },
        "emissions": {
            "s1": {"b": 1.0},
            "s2": {"b": 1.0},
        },
        "sequences": [{"id": "g", "observations": ["b"] * 40}],
    }
    model = from_dict(doc)
    tables = ModelTables(model)
    sequence = ["b"] * 40
    path, score = viterbi(tables, sequence)
    # greedy would start s2 -> all s2; global optimum is all s1
    assert path == ["s1"] * 40
    assert score == pytest.approx(math.log(0.4))


def test_ties_broken_by_full_path_lexicographic_order():
    doc = {
        "states": ["alpha", "beta"],
        "observations": ["x"],
        "initial": {"alpha": 0.5, "beta": 0.5},
        "transitions": {
            "alpha": {"alpha": 0.5, "beta": 0.5},
            "beta": {"alpha": 0.5, "beta": 0.5},
        },
        "emissions": {
            "alpha": {"x": 1.0},
            "beta": {"x": 1.0},
        },
        "sequences": [],
    }
    model = from_dict(doc)
    tables = ModelTables(model)
    path, _ = viterbi(tables, ["x", "x"])
    assert path == ["alpha", "alpha"]

    # every full path ties; lexicographically smallest complete path wins
    for length in (1, 2, 5):
        path, _ = viterbi(tables, ["x"] * length)
        assert path == ["alpha"] * length


def test_brute_force_agreement_on_randomish_model():
    rng = __import__("random").Random(1234)
    states = ["a", "b", "c"]
    symbols = ["o0", "o1"]

    def dist(names):
        values = [rng.random() for _ in names]
        total = sum(values)
        return {name: value / total for name, value in zip(names, values)}

    doc = {
        "states": states,
        "observations": symbols,
        "initial": dist(states),
        "transitions": {s: dist(states) for s in states},
        "emissions": {s: dist(symbols) for s in states},
        "sequences": [],
    }
    model = from_dict(doc)
    tables = ModelTables(model)
    sequence = [symbols[i % 2] if i % 5 else None for i in range(7)]

    best_path = None
    best_score = float("-inf")
    for path in itertools.product(states, repeat=7):
        score = math.log(model.initial[path[0]]) + math.log(
            model.emission_probability(path[0], sequence[0])
        )
        for t in range(1, 7):
            score += math.log(model.transitions[path[t - 1]][path[t]])
            score += math.log(model.emission_probability(path[t], sequence[t]))
        candidate = list(path)
        if score > best_score or (score == best_score and candidate < best_path):
            best_score = score
            best_path = candidate

    path, score = viterbi(tables, sequence)
    assert path == best_path
    assert score == pytest.approx(best_score)
    assert forward_log_likelihood(tables, sequence) == pytest.approx(
        math.log(
            sum(
                math.exp(
                    math.log(model.initial[p[0]])
                    + math.log(model.emission_probability(p[0], sequence[0]))
                    + sum(
                        math.log(model.transitions[p[t - 1]][p[t]])
                        + math.log(model.emission_probability(p[t], sequence[t]))
                        for t in range(1, 7)
                    )
                )
                for p in itertools.product(states, repeat=7)
            )
        )
    )


def test_long_sequence_does_not_underflow():
    doc = make_doc(
        transitions={
            "s1": {"s1": 0.999, "s2": 0.001},
            "s2": {"s1": 0.001, "s2": 0.999},
        },
        emissions={
            "s1": {"a": 0.999, "b": 0.001},
            "s2": {"a": 0.001, "b": 0.999},
        },
    )
    model = from_dict(doc)
    result = analyze_sequence(model, "long", ["a"] * 5000)
    assert result.status == STATUS_OK
    assert math.isfinite(result.log_likelihood)
    assert result.path == ["s1"] * 5000
    assert result.state_counts["s1"] == 5000


def test_report_preserves_order_and_contains_summary(tmp_path):
    doc = make_doc(
        sequences=[
            {"id": "one", "observations": ["a"]},
            {"id": "empty", "observations": []},
            {"id": "two", "observations": ["b"]},
        ]
    )
    path = tmp_path / "model.json"
    import json

    path.write_text(json.dumps(doc), encoding="utf-8")
    model = load_model(path)
    results = analyze_all(model)
    report = build_report(model, results)
    assert [item["id"] for item in report["results"]] == ["one", "empty", "two"]
    assert report["summary"]["total"] == 3
    assert report["summary"]["ok"] == 2
    assert report["summary"]["empty"] == 1
    assert len(report["summary"]["log_likelihoods"]) == 2
