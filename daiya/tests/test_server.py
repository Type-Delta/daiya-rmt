from __future__ import annotations

import importlib.util
import unittest


class ServerTextMessageTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipIf(importlib.util.find_spec("fastapi") is None, "fastapi is not installed")
    async def test_decoding_policy_config_is_parsed(self) -> None:
        from daiya.server import _config_from_dict

        config = _config_from_dict(
            {
                "asr_decoding_policy": "short_beam",
                "asr_short_utterance_seconds": "2.25",
            }
        )

        self.assertEqual(config.asr_decoding_policy, "short_beam")
        self.assertEqual(config.asr_short_utterance_seconds, 2.25)

    @unittest.skipIf(importlib.util.find_spec("fastapi") is None, "fastapi is not installed")
    async def test_source_message_is_acknowledged(self) -> None:
        from daiya.server import _handle_text_message

        events = await _handle_text_message('{"type":"source","source":"server-mic"}', pipeline=None)  # type: ignore[arg-type]

        self.assertEqual(
            events,
            [
                {
                    "type": "log",
                    "level": "info",
                    "source": "stream",
                    "message": "selected source: server-mic",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
