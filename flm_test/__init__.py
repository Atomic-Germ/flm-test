#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .tasks import LLMTask, EmbeddingTask, AudioTask, VisionTask, ToolCallingTask
from typing import Any


def main():
    parser = argparse.ArgumentParser(description="Test runner for FLM models.")
    parser.add_argument('--llm', action='store_true', help="Run LLM tests")
    parser.add_argument('--embedding', action='store_true', help="Run Embedding tests")
    parser.add_argument('--audio', action='store_true', help="Run Audio tests")
    parser.add_argument('--vision', action='store_true', help="Run vision tests")
    parser.add_argument('--tools', action='store_true',
                        help="Run tool-calling tests (five complexity levels)")
    parser.add_argument('--all', action='store_true', help="Run all available tests")
    parser.add_argument('--gen-lim', type=int, default=-1, help="Maximum number of tokens to generate")
    parser.add_argument('--temp', '--temperature', type=float, default=0.3, metavar='TEMP',
                        help="Sampling temperature for chat-based tests (e.g. 0.7). "
                             "Defaults to 0.3, a common setting for reliable tool calling.")
    parser.add_argument("--port", type=str, default="52625", help="Port your FLM instance is running on.")
    parser.add_argument('--backend-os', type=str, default="linux", choices=["linux", "windows"], help="OS of the FLM backend (default: linux)")
    parser.add_argument('--model', type=str, nargs='+', metavar='MODEL_ID',
                        help="Only test the specified model(s). Can be repeated or space-separated. "
                             "Example: --model gemma3:4b  or  --model gemma3:4b qwen3vl-it:4b")

    args = parser.parse_args()

    if args.all:
        args.llm = args.embedding = args.audio = args.vision = args.tools = True

    if not any([args.llm, args.embedding, args.audio, args.vision, args.tools]):
        parser.print_help()
        return

    try:
        print("Please ensure you have started the FLM server and have the correct URL and port. \n")

        host="http://127.0.0.1"
        port=str(args.port)
        endpoint="/v1"
        baseurl=f"{host}:{port}{endpoint}"

        model_filter = args.model  # list[str] | None

        if args.llm:
            LLMTask(baseurl, args.backend_os, model_filter=model_filter).run(max_completion_tokens=args.gen_lim, temperature=args.temp)

        if args.embedding:
            EmbeddingTask(baseurl, args.backend_os, model_filter=model_filter).run()

        if args.audio:
            AudioTask(baseurl, args.backend_os, model_filter=model_filter).run(temperature=args.temp)

        if args.vision:
            VisionTask(baseurl, args.backend_os, model_filter=model_filter).run(max_generation_tokens=args.gen_lim, temperature=args.temp)

        if args.tools:
            ToolCallingTask(baseurl, args.backend_os, model_filter=model_filter).run(max_completion_tokens=args.gen_lim, temperature=args.temp)

    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main()
