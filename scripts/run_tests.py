#!/usr/bin/env python3
"""
Python test runner for the project test suite.
- Ensures OPENAI_API_KEY is set (defaults to "test").
- Auto-installs pytest to the user site if missing (without changing requirements files).
- Runs API endpoint, import-structure, and rag_pipeline tests by default (the last
  auto-skips unless a real OPENAI_API_KEY is provided).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # Will be unavailable only if python-dotenv is missing


def ensure_project_root() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parent.parent
    os.chdir(project_root)
    return project_root


def load_env_file(project_root: Path) -> None:
    # Load variables from .env in project root if available
    if load_dotenv is not None:
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)


def ensure_openai_key() -> None:
    # Provide a safe default for libraries that expect a key at import-time
    os.environ.setdefault("OPENAI_API_KEY", "test")


def ensure_pytest() -> None:
    try:
        import pytest  # noqa: F401
        return
    except Exception:
        pass
    print("Installing pytest locally (user site)...")
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-q", "pytest"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Failed to install pytest.")
        sys.exit(result.returncode)


def run_pytest(targets: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-vv", "-s", *targets]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project tests")
    parser.add_argument(
        "targets",
        nargs="*",
        default=[
            "tests/test_api_endpoints.py",
            "tests/test_structure.py",
            "tests/test_rag_pipeline.py",
        ],
        help=(
            "Pytest targets (default: API, import structure, and rag_pipeline tests). "
            "Override by passing explicit targets."
        ),
    )
    return parser.parse_args()


def main() -> None:
    project_root = ensure_project_root()
    load_env_file(project_root)
    ensure_openai_key()
    ensure_pytest()
    args = parse_args()
    code = run_pytest(args.targets)
    sys.exit(code)


if __name__ == "__main__":
    main()


