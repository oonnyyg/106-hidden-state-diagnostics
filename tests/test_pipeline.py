import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hmm_diag.pipeline import analyze_document


def document():
    return {
        "model": {
            "states": ["Healthy", "Fault"],
            "observations": ["normal", "alarm"],
            "initial": {"Healthy": 0.8, "Fault": 0.2},
            "transition": {
                "Healthy": {"Healthy": 0.9, "Fault": 0.1},
                "Fault": {"Healthy": 0.0, "Fault": 1.0},
            },
            "emission": {
                "Healthy": {"normal": 1.0, "alarm": 0.0},
                "Fault": {"normal": 0.0, "alarm": 1.0},
            },
        },
        "sequences": [
            {"id": "run-ok", "observations": ["normal", None, "alarm"]},
            {"id": "run-impossible", "observations": ["alarm", "normal"]},
            {"id": "run-empty", "observations": []},
            {"id": "run-bad-symbol", "observations": ["normal", "weird"]},
        ],
    }


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.report = analyze_document(document())

    def test_order_and_summary(self):
        ids = [r["id"] for r in self.report["results"]]
        self.assertEqual(ids, ["run-ok", "run-impossible", "run-empty", "run-bad-symbol"])
        self.assertEqual(
            self.report["summary"], {"total": 4, "ok": 1, "impossible": 1, "failed": 2}
        )

    def test_ok_sequence_details(self):
        result = self.report["results"][0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["path"], ["Healthy", "Fault", "Fault"])
        self.assertLess(result["log_likelihood"], 0.0)
        self.assertEqual([t["state"] for t in result["timeline"]], result["path"])
        self.assertEqual(result["timeline"][1]["observation"], None)
        self.assertEqual(result["state_counts"], {"Healthy": 1, "Fault": 2})
        self.assertEqual(result["transition_counts"]["Healthy"]["Fault"], 1)
        self.assertEqual(result["transition_counts"]["Fault"]["Fault"], 1)

    def test_impossible_sequence(self):
        result = self.report["results"][1]
        self.assertEqual(result["status"], "impossible")
        self.assertIsNone(result["path"])
        self.assertIn("zero probability", result["reason"])

    def test_failed_sequences_have_reasons(self):
        empty, bad = self.report["results"][2], self.report["results"][3]
        self.assertEqual(empty["status"], "failed")
        self.assertIn("empty", empty["reason"])
        self.assertEqual(bad["status"], "failed")
        self.assertIn("weird", bad["reason"])


class CliTest(unittest.TestCase):
    def test_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.json"
            out = Path(tmp) / "output.json"
            src.write_text(json.dumps(document()), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "hmm_diag", str(src), "-o", str(out)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["ok"], 1)

    def test_cli_rejects_invalid_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.json"
            bad = document()
            bad["model"]["initial"] = {"Healthy": 0.5, "Fault": 0.6}
            src.write_text(json.dumps(bad), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-m", "hmm_diag", str(src)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("invalid model", proc.stderr)


if __name__ == "__main__":
    unittest.main()
