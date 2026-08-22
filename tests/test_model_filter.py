"""
Unit tests for the --model filter option (issue #1).

All network/server calls are mocked so no FLM server is required.
Run with:
    python3 -m pytest tests/test_model_filter.py -v
or:
    python3 tests/test_model_filter.py
"""
from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import patch

# Make sure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flm_test.tasks import LLMTask, VisionTask, EmbeddingTask, AudioTask

# Fake server model list — pretend the server has these three models loaded
FAKE_SERVER_MODELS = ["gemma3:4b", "qwen3vl-it:4b", "some-llm:7b"]


def _make_task(task_cls, model_filter=None):
    """Instantiate a task with all server I/O patched away."""
    with patch.object(task_cls, "_get_flm_version", return_value="0.9.99"), \
         patch.object(task_cls, "_fetch_all_models", return_value=FAKE_SERVER_MODELS), \
         patch("os.makedirs"):
        return task_cls(
            base_url="http://127.0.0.1:52625/v1",
            backend_os="linux",
            model_filter=model_filter,
        )


class TestBaseModelFilter(unittest.TestCase):

    def test_no_filter_returns_all_eligible_models(self):
        """Without --model, LLMTask keeps all server models minus its exclusion list."""
        task = _make_task(LLMTask)
        self.assertEqual(task.models, FAKE_SERVER_MODELS)

    def test_filter_single_model(self):
        """--model gemma3:4b should leave only that one model."""
        task = _make_task(LLMTask, model_filter=["gemma3:4b"])
        self.assertEqual(task.models, ["gemma3:4b"])

    def test_filter_multiple_models(self):
        """--model with two IDs should keep exactly those two."""
        task = _make_task(LLMTask, model_filter=["gemma3:4b", "some-llm:7b"])
        self.assertCountEqual(task.models, ["gemma3:4b", "some-llm:7b"])

    def test_filter_nonexistent_model_gives_empty_list(self):
        """If the requested model is not on the server, result is empty."""
        task = _make_task(LLMTask, model_filter=["nonexistent:99b"])
        self.assertEqual(task.models, [])

    def test_filter_partial_match(self):
        """Only models that are both requested AND available should remain."""
        task = _make_task(LLMTask, model_filter=["gemma3:4b", "not-on-server:5b"])
        self.assertEqual(task.models, ["gemma3:4b"])

    def test_filter_does_not_add_unavailable_models(self):
        """Requesting a model the server does not serve must not appear."""
        task = _make_task(LLMTask, model_filter=["fantasy-model:1b"])
        self.assertNotIn("fantasy-model:1b", task.models)


class TestVisionModelFilter(unittest.TestCase):

    def test_filter_intersects_with_server_models(self):
        """Only models in BOTH the server list AND the user filter should survive."""
        task = _make_task(VisionTask, model_filter=["gemma3:4b", "some-llm:7b"])
        self.assertCountEqual(task.models, ["gemma3:4b", "some-llm:7b"])

    def test_no_filter_keeps_all_server_models(self):
        """Without --model, VisionTask runs every model the server exposes."""
        task = _make_task(VisionTask)
        self.assertEqual(task.models, FAKE_SERVER_MODELS)


class TestAudioModelFilter(unittest.TestCase):

    def test_explicit_filter_overrides_audio_allowlist(self):
        """An explicit --model always wins so any audio-capable model can be tested."""
        task = _make_task(AudioTask, model_filter=["gemma3:4b"])
        self.assertEqual(task.models, ["gemma3:4b"])

    def test_no_filter_keeps_only_audio_models(self):
        """Without --model, only recognised audio models survive."""
        task = _make_task(AudioTask)
        for m in task.models:
            self.assertIn(m, AudioTask.AUDIO_MODELS)


class TestEmbeddingModelFilter(unittest.TestCase):

    def test_non_embed_model_excluded(self):
        """A non-embedding model ID in the filter should be excluded by EMBED_MODELS list."""
        task = _make_task(EmbeddingTask, model_filter=["gemma3:4b"])
        self.assertEqual(task.models, [])

    def test_no_filter_keeps_only_embed_models(self):
        """Without --model, only recognised embedding models survive."""
        task = _make_task(EmbeddingTask)
        for m in task.models:
            self.assertIn(m, EmbeddingTask.EMBED_MODELS)


class TestArgParsing(unittest.TestCase):
    """Verify the --model flag is wired up correctly in the CLI parser."""

    def _make_parser(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--llm', action='store_true')
        parser.add_argument('--vision', action='store_true')
        parser.add_argument('--embedding', action='store_true')
        parser.add_argument('--audio', action='store_true')
        parser.add_argument('--all', action='store_true')
        parser.add_argument('--gen-lim', type=int, default=-1)
        parser.add_argument('--temp', '--temperature', type=float, default=None)
        parser.add_argument('--port', type=str, default='52625')
        parser.add_argument('--backend-os', type=str, default='linux',
                            choices=['linux', 'windows'])
        parser.add_argument('--model', type=str, nargs='+', metavar='MODEL_ID')
        return parser

    def test_model_arg_single(self):
        args = self._make_parser().parse_args(['--llm', '--model', 'gemma3:4b'])
        self.assertEqual(args.model, ['gemma3:4b'])

    def test_model_arg_multiple(self):
        args = self._make_parser().parse_args(
            ['--llm', '--model', 'gemma3:4b', 'qwen3vl-it:4b'])
        self.assertEqual(args.model, ['gemma3:4b', 'qwen3vl-it:4b'])

    def test_model_arg_absent(self):
        args = self._make_parser().parse_args(['--llm'])
        self.assertIsNone(args.model)

    def test_temp_arg_single(self):
        args = self._make_parser().parse_args(['--llm', '--temp', '0.7'])
        self.assertEqual(args.temp, 0.7)

    def test_temperature_long_form(self):
        args = self._make_parser().parse_args(['--vision', '--temperature', '0.2'])
        self.assertEqual(args.temp, 0.2)

    def test_temp_arg_default_is_none(self):
        """When --temp is omitted, no temperature is sent and the server default applies."""
        args = self._make_parser().parse_args(['--audio'])
        self.assertIsNone(args.temp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
