"""Command line interface: python -m hmm_diag input.json [-o output.json]."""

from __future__ import annotations

import argparse
import json
import sys

from .model import ModelError
from .pipeline import analyze_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hmm_diag",
        description="Discrete HMM diagnostics: forward log-likelihood and Viterbi decoding.",
    )
    parser.add_argument("input", help="path to the input JSON document")
    parser.add_argument(
        "-o",
        "--output",
        help="write results to this file instead of stdout",
        default=None,
    )
    args = parser.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2

    try:
        report = analyze_document(document)
    except ModelError as exc:
        print(f"error: invalid model: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
