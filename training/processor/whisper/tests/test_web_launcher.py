from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from daiya_whisper_pipeline import web


def test_run_npm_script_treats_keyboard_interrupt_as_clean_cancellation(
   monkeypatch,
   tmp_path: Path,
) -> None:
   monkeypatch.setattr(web, "WEB_DIR", tmp_path)
   monkeypatch.setattr(web, "IS_WINDOWS", True)
   npm = tmp_path / "npm.cmd"
   npm.touch()
   npm_cli = tmp_path / "node_modules" / "npm" / "bin" / "npm-cli.js"
   npm_cli.parent.mkdir(parents=True)
   npm_cli.touch()
   monkeypatch.setattr(
      web.shutil,
      "which",
      lambda command: str(npm) if command == "npm.cmd" else "node.exe",
   )

   process = Mock()
   process.poll.return_value = None
   process.wait.side_effect = [KeyboardInterrupt, 0]
   popen = Mock(return_value=process)
   monkeypatch.setattr(web.subprocess, "Popen", popen)

   assert web.run_npm_script("dev") == web.INTERRUPTED_EXIT_CODE

   popen.assert_called_once_with(
      ["node.exe", str(npm_cli), "run", "dev"],
      cwd=tmp_path,
   )
   assert process.wait.call_count == 2
