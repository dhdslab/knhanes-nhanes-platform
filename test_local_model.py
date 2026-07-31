# -*- coding: utf-8 -*-
"""Smoke-test the local Ollama model without requiring app dependencies."""

from __future__ import annotations

import argparse
import sys

import local_llm


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the local Ollama model.")
    parser.add_argument("--model", default=local_llm.DEFAULT_MODEL)
    parser.add_argument("--url", default=local_llm.DEFAULT_URL)
    parser.add_argument(
        "--prompt",
        default="Write one concise academic sentence about epidemiologic evidence.",
    )
    args = parser.parse_args()

    print(f"URL: {args.url}")
    print(f"Model: {args.model}")

    try:
        models = local_llm.list_models(args.url)
    except RuntimeError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 2

    print("Installed models: " + (", ".join(models) if models else "(none)"))
    if args.model not in models:
        print(
            f"Model is not installed. Pull it first with: ollama pull {args.model}",
            file=sys.stderr,
        )
        return 3

    try:
        response = local_llm.generate(args.prompt, model=args.model, url=args.url, timeout=120)
    except RuntimeError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 4

    print("\nResponse:")
    print(response.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
