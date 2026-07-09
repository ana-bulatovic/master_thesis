#!/usr/bin/env python3
"""Verify project setup for the restructured thesis codebase."""

import sys
from pathlib import Path

from path_setup import PROJECT_ROOT


def main() -> int:
    print("=" * 80)
    print("NLP PIPELINE - SETUP VERIFICATION")
    print("=" * 80)

    required_paths = [
        PROJECT_ROOT / "config" / "config.yaml",
        PROJECT_ROOT / "config" / "datasets.yaml",
        PROJECT_ROOT / "src" / "pipeline" / "pipeline.py",
        PROJECT_ROOT / "src" / "data" / "dataset_loader.py",
        PROJECT_ROOT / "scripts" / "prepare_datasets.py",
        PROJECT_ROOT / "requirements.txt",
    ]

    failed = 0
    for path in required_paths:
        if path.exists():
            print(f"OK  {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"MISSING  {path.relative_to(PROJECT_ROOT)}")
            failed += 1

    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("OK  Ollama server is running")
        else:
            print("WARN  Ollama server returned non-200 status")
    except Exception:
        print("WARN  Ollama server is not running (optional for dataset prep)")

    print("\nNext steps:")
    print("  python scripts/prepare_datasets.py --dataset sarcasm")
    print("  python scripts/main.py")
    print("  python scripts/run_ui.py")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
