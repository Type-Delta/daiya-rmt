from __future__ import annotations

import unittest

from daiya.cli import build_parser


class CLIConfigTests(unittest.TestCase):
    def test_decoding_policy_defaults_are_backward_compatible(self) -> None:
        args = build_parser().parse_args(["sample.wav"])

        self.assertEqual(args.asr_decoding_policy, "baseline")
        self.assertEqual(args.asr_short_utterance_seconds, 3.0)

    def test_named_decoding_policy_and_threshold_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "sample.wav",
                "--asr-decoding-policy",
                "short_greedy",
                "--asr-short-utterance-seconds",
                "1.75",
            ]
        )

        self.assertEqual(args.asr_decoding_policy, "short_greedy")
        self.assertEqual(args.asr_short_utterance_seconds, 1.75)


if __name__ == "__main__":
    unittest.main()
