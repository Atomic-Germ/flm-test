import base64
import csv
import os
import re
import time
import subprocess
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# Media assets ship inside the package; anchor paths here so tests work
# regardless of the current working directory (repo checkout or pip/uv install).
PACKAGE_DIR = Path(__file__).resolve().parent

class BaseTestTask(ABC):
    """
    Abstract base class for all testing tasks.
    Enforces a standard interface for running tests and saving results.
    """
    MUSIC_PATTERN = re.compile(
        r"\b(music|melod\w*|song|tune|rhythm|beat|tempo|instrument\w*|drum\w*|bass|synth\w*|vocal\w*|chord\w*|harmo\w*)\b",
        re.IGNORECASE,
    )

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="flm")
        self.version = self._get_flm_version()
        self.models = self._fetch_all_models()
        if model_filter:
            filtered = [m for m in self.models if m in model_filter]
            if not filtered:
                print(f"Warning: none of the requested model(s) {model_filter} were found in the server's model list. "
                      f"Available: {self.models}")
            self.models = filtered
        self.results_dir = os.path.join("results", self.timestamp, backend_os)
        os.makedirs(self.results_dir, exist_ok=True)

    def get_csv_filename(self, task_name: str) -> str:
        return os.path.join(self.results_dir, f"{task_name}_results_v{self.version}.csv")

    def _get_flm_version(self) -> str:
        print("\nChecking flm version...")
        try:
            response = urllib.request.urlopen(f"{self.base_url}/version", timeout=5)
            version_data = json.loads(response.read().decode('utf-8'))
            flm_version = version_data.get("version", "unknown_version")
            print(f"Detected flm version: {flm_version}")
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"Error fetching flm version: {e}")
            flm_version = "unknown_version"
        return flm_version

    def _fetch_all_models(self) -> list:
        print("\nFetching available models...")
        try:
            response = urllib.request.urlopen(f"{self.base_url}/models", timeout=5)
            models_json = json.loads(response.read().decode('utf-8'))
            model_list = models_json.get("data", [])
            model_id = [m["id"] for m in model_list]
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            print(f"Error fetching models: {e}")
            model_id = []
        return model_id

    def _collect_stream(self, response) -> tuple[str, str]:
        """Accumulates streamed chunks into (reasoning_content, output_content)."""
        reasoning_content, output_content = "", ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            if delta.content:
                output_content += delta.content
        return reasoning_content, output_content

    def start_flm_server(self, audio, embed):
        """Starts the flm server as a subprocess."""
        print("Starting flm server...")
        server_process = subprocess.Popen(["flm", "serve", "-a", str(audio), "-e", str(embed)])
        time.sleep(5) # Allow server to boot
        return server_process

    @abstractmethod
    def run(self, *args, **kwargs):
        """Must be implemented by all subclasses"""
        pass


class LLMTask(BaseTestTask):

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.models = [m for m in self.models if m not in ("gpt-oss:20b", "gpt-oss-sg:20b", "qwen3.5:4b", "qwen3.5:9b", "medgemma:4b", "medgemma1.5:4b", "translategemma:4b")]
        self.csv_filename = self.get_csv_filename("llm")

    def _run_two_rounds(self, writer, model_id, prompt, followup_prompt, stream, max_completion_tokens, temperature=None):
        mode = "Stream" if stream else "Non-Stream"
        messages = [{"role": "user", "content": prompt}]

        # first round
        try:
            print(f"Prompt: {prompt}")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=stream,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            if stream:
                reasoning_content, output_content = self._collect_stream(response)
            else:
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", "N/A") or "N/A"
                output_content = response.choices[0].message.content or ""
            writer.writerow([model_id, mode, prompt, reasoning_content or "N/A", output_content])
            print("Done.")
            time.sleep(1)
            messages.append({"role": "assistant", "content": output_content})
            messages.append({"role": "user", "content": followup_prompt})
        except Exception as e:
            print(f"Error occurred in first round, model: {model_id}: {e}")
            writer.writerow([model_id, mode, prompt, f"ERROR: {e}", "N/A"])

        # second round
        try:
            print(f"Follow-up Prompt: {followup_prompt}")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=stream,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            if stream:
                reasoning_content, output_content = self._collect_stream(response)
            else:
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", "N/A") or "N/A"
                output_content = response.choices[0].message.content or ""
            writer.writerow([model_id, mode, followup_prompt, reasoning_content or "N/A", output_content])
            print("Done.")
            time.sleep(1)
        except Exception as e:
            print(f"Error occurred in second round, model: {model_id}: {e}")
            writer.writerow([model_id, mode, followup_prompt, f"ERROR: {e}", "N/A"])

    def run(self, max_completion_tokens=-1, temperature=None):
        prompt = "Teach me Maxwell's equations."
        followup_prompt = "Summarize your answer."

        stream_prompt = "Teach me Maxwell's equations."
        stream_followup_prompt = "Explain why they are important."

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Mode", "Input", "Reasoning Content", "Output Content"])
            print("\n=== Starting LLM Tests ===")
            print(f"Models found: {len(self.models)}")
            for model_id in self.models:
            # for model_id in self.models[2:4]:  # Limit to first 2 models for testing purposes
                print(f"\n--- Testing LLM model: {model_id} ---")
                # print("Testing non-stream mode...\n")
                # self._run_two_rounds(writer, model_id, prompt, followup_prompt, stream=False, max_completion_tokens=max_completion_tokens)
                print("\nTesting stream mode...\n")
                self._run_two_rounds(writer, model_id, stream_prompt, stream_followup_prompt, stream=True, max_completion_tokens=max_completion_tokens, temperature=temperature)
                print(f"Finished testing model: {model_id}")
        print(f"\nLLM tests complete. Saved to {self.csv_filename}")


class EmbeddingTask(BaseTestTask):
    EMBED_MODELS = ["embed-gemma:300m"]

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        # Keep only recognised embedding models; honour any user-supplied filter.
        self.models = [m for m in self.models if m in self.EMBED_MODELS]
        self.csv_filename = self.get_csv_filename("embedding")

    def run(self):
        print("\n=== Starting Embedding Tests ===")
        print("Testing the following Embedding models:")
        for i, model in enumerate(self.models, 1):
            print(f"  {i}. {model}")

        server_process = self.start_flm_server(audio=0, embed=1)

        # TODO: Implement OpenAI Embeddings API calls
        # client.embeddings.create(input="text", model=model_id)

        print("\nShutting down flm server...")
        server_process.terminate()
        server_process.wait()
        print(f"Embedding tests complete. Saved to {self.csv_filename}")

class AudioTask(BaseTestTask):
    AUDIO_MODELS = ["whisper-v3:turbo"]

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        # Without an explicit --model filter, keep only recognised audio models;
        # an explicit filter always wins so any audio-capable model can be tested.
        if not model_filter:
            self.models = [m for m in self.models if m in self.AUDIO_MODELS]
        self.test_audio_path = PACKAGE_DIR / "test_files" / "audio" / "atomic-germ.mp3"
        self.csv_filename = self.get_csv_filename("audio")

    def _load_audio_base64(self, audio_path) -> str:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')

    def run(self, max_generation_tokens=-1, temperature=None):
        prompt = "Describe what you hear in this audio clip."
        followup_prompt = "What kind of mood or genre would this clip fit into?"

        audio_b64 = self._load_audio_base64(self.test_audio_path)

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Input", "Reasoning Content", "Output Content", "Music Mention Check"])
            print("\n=== Starting Audio Tests ===")
            print(f"Models found: {len(self.models)}")
            for i, model_id in enumerate(self.models, 1):
                print(f"\n--- Testing audio models ({i}/{len(self.models)}): {model_id} ---")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                        ],
                    }
                ]

                # first round
                try:
                    print(f"Prompt: {prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    music_check = "PASS" if self.MUSIC_PATTERN.search(output_content) else "SOFT-FAIL"
                    writer.writerow([model_id, prompt, reasoning_content or "N/A", output_content, music_check])
                    if music_check == "PASS":
                        print("Music mention check: PASS")
                    else:
                        print("Music mention check: SOFT-FAIL (noted only, not a hard failure)")
                    print("Done.")
                    time.sleep(1)
                    messages.append({"role": "assistant", "content": output_content})
                    messages.append({"role": "user", "content": followup_prompt})
                except Exception as e:
                    print(f"Error occurred in first round, model: {model_id}: {e}")
                    writer.writerow([model_id, prompt, f"ERROR: {e}", "N/A", "ERROR"])

                # second round
                try:
                    print(f"Follow-up Prompt: {followup_prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    writer.writerow([model_id, followup_prompt, reasoning_content or "N/A", output_content, "N/A"])
                    print("Done.")
                    time.sleep(1)
                except Exception as e:
                    print(f"Error occurred in second round, model: {model_id}: {e}")
                    writer.writerow([model_id, followup_prompt, f"ERROR: {e}", "N/A", "N/A"])
                print(f"Finished testing model: {model_id}")
        print(f"Audio tests complete. Saved to {self.csv_filename}")

class VisionTask(BaseTestTask):

    EXPECTED_TEXT = ("The capital of France is Paris. It is a major global city "
                     "and serves as the nation's center for finance, commerce, "
                     "culture, arts, fashion, and science.")
    SEAGULL_PATTERN = re.compile(r"\b(?:sea[\s-]?)?gulls?\b", re.IGNORECASE)

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.test_image1_path = PACKAGE_DIR / "test_files" / "image" / "paris.png"
        self.test_image2_path = PACKAGE_DIR / "test_files" / "image" / "seagull.jpeg"
        self.test_image3_path = PACKAGE_DIR / "test_files" / "image" / "spectrogram.png"
        self.csv_filename = self.get_csv_filename("vision")
        # Tolerant regex: case-insensitive, flexible whitespace, optional apostrophe.
        escaped_words = [re.escape(word) for word in self.EXPECTED_TEXT.split()]
        escaped_words = [w.replace(r"\'", "'?").replace("'", "'?") for w in escaped_words]
        self.expected_text_pattern = re.compile(
            r"\s+".join(escaped_words),
            re.IGNORECASE,
        )
        for i, model in enumerate(self.models, 1):
            print(f"  {i}. {model}")

    def _load_image_base64(self, image_path) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def run(self, max_generation_tokens=-1, temperature=None):
        prompt = "Extract text from the first image, describe the second one, and imagine what the spectrogram might sound like."
        followup_prompt = "Make a story that connects the images together."
        followup_prompt_music = "What kind of sound does the spectrogram represent?"

        image1_b64 = self._load_image_base64(self.test_image1_path)
        image2_b64 = self._load_image_base64(self.test_image2_path)
        image3_b64 = self._load_image_base64(self.test_image3_path)

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Input", "Reasoning Content", "Output Content",
                             "Text Extraction Check", "Seagull Mention Check", "Spectrogram Music Check"])
            print("\n=== Starting Vision Tests ===")
            print(f"Models found: {len(self.models)}")
            for model_id in self.models:
                print(f"\n--- Testing VLMs: {model_id} ---")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image1_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image2_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image3_b64}"}},
                        ],
                    }
                ]

                # first round
                seagull_in_description = "ERROR"
                music_in_description = "ERROR"
                try:
                    print(f"Prompt: {prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    text_check = "PASS" if self.expected_text_pattern.search(output_content) else "FAIL"
                    if text_check == "PASS":
                        print("Text extraction check: PASS")
                    else:
                        print("Text extraction check: FAIL (expected text not found in response)")
                    seagull_in_description = "PASS" if self.SEAGULL_PATTERN.search(output_content) else "FAIL"
                    music_in_description = "PASS" if self.MUSIC_PATTERN.search(output_content) else "FAIL"
                    writer.writerow([model_id, prompt, reasoning_content or "N/A", output_content,
                                     text_check, seagull_in_description, music_in_description])
                    print("Done.")
                    time.sleep(1)
                    messages.append({"role": "assistant", "content": output_content})
                    messages.append({"role": "user", "content": followup_prompt})
                except Exception as e:
                    print(f"Error occurred in first round, model: {model_id}: {e}")
                    writer.writerow([model_id, prompt, f"ERROR: {e}", "N/A", "ERROR", "ERROR", "ERROR"])

                # second round
                round2_ok = False
                try:
                    print(f"Follow-up Prompt: {followup_prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    seagull_in_story = "PASS" if self.SEAGULL_PATTERN.search(output_content) else "FAIL"
                    writer.writerow([model_id, followup_prompt, reasoning_content or "N/A", output_content,
                                     "N/A", seagull_in_story, "N/A"])
                    if seagull_in_story == "PASS":
                        print("Seagull check (story): PASS")
                    elif seagull_in_description == "PASS":
                        print("Seagull check: PASS (recognized in description)")
                    else:
                        print("Seagull check: FAIL (no mention of a seagull in story or description)")
                    print("Done.")
                    time.sleep(1)
                    messages.append({"role": "assistant", "content": output_content})
                    messages.append({"role": "user", "content": followup_prompt_music})
                    round2_ok = True
                except Exception as e:
                    print(f"Error occurred in second round, model: {model_id}: {e}")
                    writer.writerow([model_id, followup_prompt, f"ERROR: {e}", "N/A", "N/A", "ERROR", "N/A"])

                # third round (spectrogram; informational, not a hard failure)
                music_check = "SKIPPED"
                if round2_ok:
                    try:
                        print(f"Follow-up Prompt: {followup_prompt_music}")
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=messages,
                            stream=True,
                            max_completion_tokens=max_generation_tokens,
                            temperature=temperature,
                        )
                        reasoning_content, output_content = self._collect_stream(response)
                        music_check = "PASS" if self.MUSIC_PATTERN.search(output_content) else "SOFT-FAIL"
                        writer.writerow([model_id, followup_prompt_music, reasoning_content or "N/A",
                                         output_content, "N/A", "N/A", music_check])
                        if music_check == "PASS":
                            print("Spectrogram music check: PASS")
                        elif music_in_description == "PASS":
                            print("Spectrogram music check: PASS (recognized in description)")
                        else:
                            print("Spectrogram music check: SOFT-FAIL (noted only, not a hard failure)")
                        print("Done.")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error occurred in third round, model: {model_id}: {e}")
                        writer.writerow([model_id, followup_prompt_music, f"ERROR: {e}", "N/A",
                                         "N/A", "N/A", "ERROR"])
                else:
                    writer.writerow([model_id, followup_prompt_music,
                                     "SKIPPED: second round failed", "N/A", "N/A", "N/A", "SKIPPED"])
                print(f"Finished testing model: {model_id}")
        print(f"Vision tests complete. Saved to {self.csv_filename}")
