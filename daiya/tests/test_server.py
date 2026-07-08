from __future__ import annotations

import importlib.util
import unittest


class ServerTextMessageTests(unittest.IsolatedAsyncioTestCase):
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


class ServerConfigTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("fastapi") is None, "fastapi is not installed")
    def test_segmenter_fields_are_parsed_from_dict(self) -> None:
        from daiya.server import _config_from_dict

        config = _config_from_dict(
            {
                "segmenter_backend": "auto",
                "vad_threshold": "0.42",
                "vad_min_speech_seconds": "0.11",
                "vad_min_silence_seconds": "0.33",
                "vad_speech_padding_seconds": "0.07",
                "utterance_cap_seconds": "3.5",
            }
        )

        self.assertEqual(config.segmenter_backend, "auto")
        self.assertEqual(config.vad_threshold, 0.42)
        self.assertEqual(config.vad_min_speech_seconds, 0.11)
        self.assertEqual(config.vad_min_silence_seconds, 0.33)
        self.assertEqual(config.vad_speech_padding_seconds, 0.07)
        self.assertEqual(config.utterance_cap_seconds, 3.5)


if __name__ == "__main__":
    unittest.main()
