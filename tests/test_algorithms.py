import math
import unittest

from hmm_diag.algorithms import forward_log_likelihood, viterbi
from hmm_diag.model import HMMModel


def make_model():
    return HMMModel.from_dict(
        {
            "states": ["Healthy", "Fault"],
            "observations": ["normal", "alarm"],
            "initial": {"Healthy": 0.8, "Fault": 0.2},
            "transition": {
                "Healthy": {"Healthy": 0.9, "Fault": 0.1},
                "Fault": {"Healthy": 0.3, "Fault": 0.7},
            },
            "emission": {
                "Healthy": {"normal": 0.9, "alarm": 0.1},
                "Fault": {"normal": 0.2, "alarm": 0.8},
            },
        }
    )


def brute_force_log_likelihood(model, observations):
    def emit(state, obs):
        return 1.0 if obs is None else model.emission[state][obs]

    total = 0.0
    paths = [[]]
    for obs in observations:
        paths = [p + [s] for p in paths for s in model.states]
    for path in paths:
        prob = model.initial[path[0]] * emit(path[0], observations[0])
        for prev, cur, obs in zip(path, path[1:], observations[1:]):
            prob *= model.transition[prev][cur] * emit(cur, obs)
        total += prob
    return math.log(total)


class ForwardTest(unittest.TestCase):
    def test_matches_brute_force(self):
        model = make_model()
        observations = ["normal", None, "alarm", "normal"]
        self.assertAlmostEqual(
            forward_log_likelihood(model, observations),
            brute_force_log_likelihood(model, observations),
            places=12,
        )

    def test_empty_sequence_rejected(self):
        model = make_model()
        with self.assertRaises(ValueError):
            forward_log_likelihood(model, [])
        with self.assertRaises(ValueError):
            viterbi(model, [])

    def test_long_sequence_no_underflow(self):
        model = make_model()
        observations = ["normal", "alarm"] * 500
        log_lik = forward_log_likelihood(model, observations)
        self.assertTrue(math.isfinite(log_lik))
        self.assertLess(log_lik, 0.0)
        decoded = viterbi(model, observations)
        self.assertIsNotNone(decoded)
        self.assertEqual(len(decoded[0]), 1000)


class ViterbiTest(unittest.TestCase):
    def test_global_not_greedy(self):
        # Greedy per-step choice would pick Healthy at t=0 (0.8 vs 0.2),
        # but the alarm run makes the joint Fault path globally optimal.
        model = make_model()
        path, _ = viterbi(model, ["alarm", "alarm", "alarm"])
        self.assertEqual(path, ["Fault", "Fault", "Fault"])

    def test_tie_break_lexicographic(self):
        model = HMMModel.from_dict(
            {
                "states": ["Beta", "Alpha"],
                "observations": ["x"],
                "initial": {"Beta": 0.5, "Alpha": 0.5},
                "transition": {
                    "Beta": {"Beta": 0.5, "Alpha": 0.5},
                    "Alpha": {"Beta": 0.5, "Alpha": 0.5},
                },
                "emission": {
                    "Beta": {"x": 1.0},
                    "Alpha": {"x": 1.0},
                },
            }
        )
        path, _ = viterbi(model, ["x", "x"])
        self.assertEqual(path, ["Alpha", "Alpha"])

    def test_missing_observation_still_transitions(self):
        model = HMMModel.from_dict(
            {
                "states": ["A", "B"],
                "observations": ["x"],
                "initial": {"A": 1.0, "B": 0.0},
                "transition": {
                    "A": {"A": 0.0, "B": 1.0},
                    "B": {"A": 0.0, "B": 1.0},
                },
                "emission": {
                    "A": {"x": 1.0},
                    "B": {"x": 1.0},
                },
            }
        )
        path, _ = viterbi(model, [None, None, None])
        self.assertEqual(path, ["A", "B", "B"])

    def test_impossible_sequence_returns_none(self):
        model = HMMModel.from_dict(
            {
                "states": ["A"],
                "observations": ["x", "y"],
                "initial": {"A": 1.0},
                "transition": {"A": {"A": 1.0}},
                "emission": {"A": {"x": 1.0, "y": 0.0}},
            }
        )
        self.assertIsNone(viterbi(model, ["x", "y"]))
        self.assertEqual(forward_log_likelihood(model, ["x", "y"]), float("-inf"))


if __name__ == "__main__":
    unittest.main()
