"""Command line entry point: ``python -m hmm_diagnostics model.json``."""

from __future__ import annotations

import argparse
import json
import sys

from .diagnose import analyze_all, build_report
from .model import HMMValidationError, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hmm_diagnostics",
        description="Analyze labeled observation sequences with a discrete HMM.",
    )
    parser.add_argument("model", help="path to the JSON model file")
    parser.add_argument("-o", "--output", help="write JSON report to this path")
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indentation for the report (default: 2)"
    )
    args = parser.parse_args(argv)

    try:
        model = load_model(args.model)
    except (OSError, json.JSONDecodeError, HMMValidationError) as exc:
        print(f"error: failed to load model: {exc}", file=sys.stderr)
        return 2

    results = analyze_all(model)
    report = build_report(model, results)
    text = json.dumps(report, ensure_ascii=False, indent=args.indent, allow_nan=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)

    failed = report["summary"]["total"] - report["summary"]["ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
