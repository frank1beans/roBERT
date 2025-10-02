#!/usr/bin/env python3
"""Utility to verify the repository has the expected structure and passes tests."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REQUIRED_PATHS = [
    Path("pyproject.toml"),
    Path("src"),
    Path("src/robimb"),
    Path("tests"),
    Path("docs"),
    Path("data"),
]


def find_missing(paths: Iterable[Path]) -> list[Path]:
    """Return the paths that are not present in the repository."""
    return [path for path in paths if not path.exists()]


def run_pytest(pytest_args: list[str] | None = None) -> int:
    """Execute pytest with the provided arguments and return its exit code."""
    cmd = [sys.executable, "-m", "pytest"]
    if pytest_args:
        cmd.extend(pytest_args)
    process = subprocess.run(cmd, check=False)
    return process.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running the pytest suite.",
    )
    parser.add_argument(
        "pytest-args",
        nargs=argparse.REMAINDER,
        help="Arguments passed directly to pytest (must come after --).",
    )
    args = parser.parse_args(argv)

    missing = find_missing(REQUIRED_PATHS)
    if missing:
        print("Missing required paths:")
        for path in missing:
            print(f" - {path}")
    else:
        print("All required paths are present.")

    exit_code = 0
    if not args.skip_tests:
        pytest_args = args.__dict__.get("pytest-args") or []
        if pytest_args and pytest_args[0] == "--":
            pytest_args = pytest_args[1:]
        print("\nRunning pytest...")
        exit_code = run_pytest(pytest_args)
        if exit_code == 0:
            print("Pytest completed successfully.")
        else:
            print(f"Pytest exited with code {exit_code}.")

    if missing and exit_code == 0:
        return 1
    if missing or exit_code != 0:
        return exit_code or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
