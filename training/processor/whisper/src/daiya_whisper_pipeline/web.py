from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


WEB_DIR = Path(__file__).resolve().parents[2] / "web"
INTERRUPTED_EXIT_CODE = 130
INTERRUPT_GRACE_SECONDS = 5
IS_WINDOWS = os.name == "nt"


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
   if process.poll() is not None:
      return

   try:
      process.wait(timeout=INTERRUPT_GRACE_SECONDS)
      return
   except subprocess.TimeoutExpired:
      if IS_WINDOWS:
         subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
         )
      else:
         process.terminate()

   try:
      process.wait(timeout=INTERRUPT_GRACE_SECONDS)
   except subprocess.TimeoutExpired:
      process.kill()
      process.wait()


def _npm_command(npm: str, script: str) -> list[str]:
   launcher = [npm]
   if IS_WINDOWS:
      npm_cli = (
         Path(npm).resolve().parent
         / "node_modules"
         / "npm"
         / "bin"
         / "npm-cli.js"
      )
      node = shutil.which("node.exe") or shutil.which("node")
      if node is not None and npm_cli.is_file():
         launcher = [node, str(npm_cli)]

   if script in ("install", "i"):
      return [*launcher, script]
   return [*launcher, "run", script]


def build_parser() -> argparse.ArgumentParser:
   parser = argparse.ArgumentParser(
      prog="daiya-wpl-web",
      description="Run Daiya Whisper Pipeline web scripts from the repo root.",
   )
   parser.add_argument(
      "script",
      choices=("build", "dev", "start", "preview", "install", "i"),
      nargs="?",
      default="build",
      help="npm script to run in training/processor/whisper/web. Defaults to build.",
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
      raise SystemExit(
         "npm was not found on PATH; install Node.js/npm to build the web UI.")
   if not WEB_DIR.exists():
      raise SystemExit(f"web directory does not exist: {WEB_DIR}")

   command = _npm_command(npm, script)
   if extra_args:
      command.extend(["--", *extra_args])

   process = subprocess.Popen(command, cwd=WEB_DIR)
   try:
      return process.wait()
   except KeyboardInterrupt:
      _stop_process_tree(process)
      return INTERRUPTED_EXIT_CODE


def main() -> None:
   args = build_parser().parse_args()
   raise SystemExit(run_npm_script(args.script, args.args))


def build_main() -> None:
   raise SystemExit(run_npm_script("build"))


if __name__ == "__main__":
   main()
