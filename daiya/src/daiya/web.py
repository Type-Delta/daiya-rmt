from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daiya-web",
        description="Run Daiya web package scripts from the repo root.",
    )
    parser.add_argument(
        "script",
        choices=("build", "dev", "preview"),
        nargs="?",
        default="build",
        help="npm script to run in daiya/web. Defaults to build.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after npm's -- separator.",
    )
    return parser


def run_npm_script(script: str, extra_args: list[str] | None = None) -> int:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise SystemExit("npm was not found on PATH; install Node.js/npm to build the web UI.")
    if not WEB_DIR.exists():
        raise SystemExit(f"web directory does not exist: {WEB_DIR}")

    command = [npm, "run", script]
    if extra_args:
        command.extend(["--", *extra_args])
    return subprocess.run(command, cwd=WEB_DIR).returncode


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_npm_script(args.script, args.args))


def build_main() -> None:
    raise SystemExit(run_npm_script("build"))


if __name__ == "__main__":
    main()
