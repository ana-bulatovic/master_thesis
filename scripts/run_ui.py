#!/usr/bin/env python3
"""Launch Streamlit UI for the thesis pipeline."""

import subprocess
import sys
from pathlib import Path

from path_setup import PROJECT_ROOT


def main():
    app_path = PROJECT_ROOT / "ui" / "app.py"
    if not app_path.exists():
        print(f"Error: UI app not found at {app_path}")
        return 1

    print("Launching NLP Pipeline Web UI...")
    print("Open http://localhost:8501 in your browser")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
