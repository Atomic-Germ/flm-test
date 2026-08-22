# flm-test

A comprehensive testing framework *intended* for  **[FastFlowLM (FLM)](https://fastflowlm.com)** that validates the functionality of various AI model categories including Language, Embedding, Audio, and Vision models.

## Overview

flm-test is designed to thoroughly test FastFlowLM's API compatibility and model functionality across multiple modalities:

- **LLM Tests**: Language model inference with both streaming and non-streaming modes
- **Embedding Tests**: Text embedding model validation
- **Audio Tests**: Audio understanding via chat completions, with a bundled music clip
- **Vision Tests**: Vision-Language Model (VLM) tests with multi-image support and automated response checking

All test media is **bundled inside the package**, so no extra downloads or local paths are needed once installed.

Each test suite automatically:
- Detects the FLM server version
- Fetches available models
- Runs standardized test prompts against the bundled media
- Applies pass/fail response checks where applicable
- Saves results to CSV with timestamps
- Handles errors gracefully with detailed logging

## Prerequisites

- **FastFlowLM server** running locally or remotely
- **`uv` or `pip`** (Python package manager)

## Quick Start

### 1. Install the package

```bash
uv pip install git+https://github.com/Atomic-Germ/flm-test.git
```
```bash
pip install git+https://github.com/Atomic-Germ/flm-test.git
```

Or as an isolated tool with `uv`:

```bash
uv tool install git+https://github.com/Atomic-Germ/flm-test.git
```

> Note: PyPI hosting is planned; until then, install directly from GitHub.

### 2. Start FLM Server

Ensure your FastFlowLM server is running before running tests. Start the server with appropriate flags based on the tests you plan to run:

**Basic local server:**
```bash
flm serve
```

**Load embedding models (required for embedding tests):**
```bash
flm serve -e 1
```

### 3. Run Tests

Run tests with:

```bash
# Run all tests
flm-test --all

# Run specific tests
flm-test --llm                    # LLM tests only
flm-test --embedding              # Embedding tests only
flm-test --audio                  # Audio tests only
flm-test --vision                 # Vision tests only

# Target a specific model (instead of all available models)
flm-test --llm --model gemma3:4b
flm-test --vision --model gemma3:4b qwen3vl-it:4b   # space-separated list
flm-test --audio --model whisper-v3:turbo

# Configuration
flm-test --llm --port 56354       # Set a custom port for LFM
flm-test --llm --gen-lim 32       # Limit LLM output to 32 tokens
```

## Test Types

### LLM Tests
Tests language models with conversation capabilities.

**What it tests:**
- Non-streaming mode: Single API calls with standard responses
- Streaming mode: Continuous token-by-token responses
- Multi-turn conversations: Context preservation across exchanges
- Reasoning content extraction (if supported by model)

**Test Flow:**

**Non-stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Summarize your answer."

**Stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Explain why they are important."

**Output:** `llm_results_v{version}_{timestamp}.csv`

### Vision Tests
Tests Vision-Language Models (VLMs) with multi-image analysis and objective response validation.

**What it tests:**
- OCR/text extraction from an image
- Multi-image understanding and detailed description generation
- Creative story generation connecting multiple images
- Streaming responses for image-to-text

**Test Flow:**
1. Initial prompt: "Extract text from the first image, describe the second one, and imagine what the spectrogram might sound like."
2. Follow-up: "Make a story that connects the images together."
3. Follow-up: "What kind of sound does the spectrogram represent?"

**Bundled Test Media:**
- `test_files/image/paris.png` - image containing a known English sentence
- `test_files/image/seagull.jpeg` - photograph of a seagull on a lamp post
- `test_files/image/spectrogram.png` - spectrogram of a musical clip

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| Text Extraction Check | PASS / FAIL | The first-round response must contain the exact sentence shown in `paris.png`, matched case-insensitively with flexible whitespace |
| Seagull Mention Check | PASS / FAIL | A seagull ("seagull", "sea gull", or "gull") must be recognized in the description or the story |
| Spectrogram Music Check | PASS / SOFT-FAIL | Informational only: the response should reference music-related terms (melody, rhythm, instruments, etc.). Failure is noted but does not count as a hard failure |

**Output:** `vision_results_v{version}_{timestamp}.csv`

### Audio Tests
Tests audio-capable models through chat completions using the OpenAI-style `input_audio` content part.

**What it tests:**
- Sending base64-encoded MP3 audio inline in a chat request
- Audio comprehension and description quality
- Multi-turn context preservation after an audio exchange
- Reasoning content extraction (if supported by model)

**Test Flow:**
1. Initial prompt: "Describe what you hear in this audio clip."
2. Follow-up: "What kind of mood or genre would this clip fit into?"

**Bundled Test Media:**
- `test_files/audio/atomic-germ.mp3` - short instrumental music clip (~64 seconds)

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| Music Mention Check | PASS / SOFT-FAIL | The description should reference music-related terms (melody, rhythm, beat, instrument, etc.). Failure is noted but does not count as a hard failure |

By default, audio tests run against known audio models (currently `whisper-v3:turbo`). An explicit `--model` filter always wins, so any audio-capable model can be targeted directly.

**Output:** `audio_results_v{version}_{timestamp}.csv`

## Understanding Results

Test results are saved as CSV files under timestamped directories:
```
results/{timestamp}/{backend_os}/{test_type}_results_v{flm_version}.csv
```

**Example filenames:**
- `results/20260821_203124/linux/vision_results_v1.0.1.csv`

### CSV Columns

**LLM Results:**
| Column | Description |
|--------|-------------|
| Model | Model ID/name |
| Mode | "Stream" or "Non-Stream" |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |

**Vision Results:**
| Column | Description |
|--------|-------------|
| Model | VLM model ID |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |
| Text Extraction Check | PASS / FAIL / ERROR for the paris.png sentence |
| Seagull Mention Check | PASS / FAIL / ERROR per round (description and story) |
| Spectrogram Music Check | PASS / SOFT-FAIL / ERROR / SKIPPED |

**Audio Results:**
| Column | Description |
|--------|-------------|
| Model | Audio model ID |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |
| Music Mention Check | PASS / SOFT-FAIL / ERROR |

### Interpreting Results

- **N/A**: Feature/check not applicable to that row
- **PASS**: Response satisfied the check
- **FAIL**: Hard requirement not met (text extraction, seagull recognition)
- **SOFT-FAIL**: Noted for review only; not counted as a hard failure
- **ERROR: {message}**: Test failed with specific error
- **SKIPPED**: Round skipped due to an earlier failure
- **Empty content**: Model timeout or connection issue
