import unittest

from hmm_diag.model import HMMModel, ModelError


def valid_model_dict():
    return {
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


class ModelValidationTest(unittest.TestCase):
    def test_valid_model_loads(self):
        model = HMMModel.from_dict(valid_model_dict())
        self.assertEqual(model.states, ["Healthy", "Fault"])

    def test_duplicate_states_rejected(self):
        raw = valid_model_dict()
        raw["states"] = ["Healthy", "Healthy"]
        with self.assertRaises(ModelError):
            HMMModel.from_dict(raw)

    def test_duplicate_observations_rejected(self):
        raw = valid_model_dict()
        raw["observations"] = ["alarm", "alarm"]
        with self.assertRaises(ModelError):
            HMMModel.from_dict(raw)

    def test_distribution_sum_tolerance(self):
        raw = valid_model_dict()
        raw["initial"] = {"Healthy": 0.8 + 2e-12, "Fault": 0.2}
        with self.assertRaises(ModelError):
            HMMModel.from_dict(raw)
        raw["initial"] = {"Healthy": 0.8 + 5e-13, "Fault": 0.2}
        HMMModel.from_dict(raw)

    def test_non_finite_and_out_of_range_rejected(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1, "0.5", True):
            raw = valid_model_dict()
            raw["initial"] = {"Healthy": bad, "Fault": 0.2}
            with self.assertRaises(ModelError, msg=repr(bad)):
                HMMModel.from_dict(raw)

    def test_zero_probability_allowed(self):
        raw = valid_model_dict()
        raw["initial"] = {"Healthy": 1.0, "Fault": 0.0}
        model = HMMModel.from_dict(raw)
        self.assertEqual(model.initial["Fault"], 0.0)

    def test_unknown_and_missing_entries_rejected(self):
        raw = valid_model_dict()
        raw["transition"]["Healthy"] = {"Healthy": 0.5, "Ghost": 0.5}
        with self.assertRaises(ModelError):
            HMMModel.from_dict(raw)
        raw = valid_model_dict()
        del raw["emission"]["Fault"]["alarm"]
        raw["emission"]["Fault"]["normal"] = 1.0
        with self.assertRaises(ModelError):
            HMMModel.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
