# flm-test

A comprehensive testing framework for **[FastFlowLM (FLM)](https://fastflowlm.com)** that validates the functionality of various AI model categories including Language, Embedding, Audio, and Vision models.

## Overview

flm-test is designed to thoroughly test FastFlowLM's API compatibility and model functionality across multiple modalities:

- **LLM Tests**: Language model inference with both streaming and non-streaming modes
- **Embedding Tests**: Text embedding model validation
- **Audio Tests**: Speech recognition and audio processing validation  
- **Vision Tests**: Vision-Language Model (VLM) tests with multi-image support

Each test suite automatically:
- Detects the FLM server version
- Fetches available models
- Runs standardized test prompts
- Saves results to CSV with timestamps
- Handles errors gracefully with detailed logging

## Prerequisites

- **FastFlowLM server** running locally or remotely
- **`uv` or `pip`** (Python package manager)

## Quick Start

### 1. Install the package

```bash
uv tool install git+https://github.com/Atomic-Germ/flm-test
```
```bash
pip install git+https://github.com/Atomic-Germ/flm-test
```


### 4. Start FLM Server

Ensure your FastFlowLM server is running before running tests. Start the server with appropriate flags based on the tests you plan to run:

**Basic local server:**
```bash
flm serve
```

**Load embedding models (required for embedding tests):**
```bash
flm serve -e 1
```

**Load audio models (required for audio tests):**
```bash
flm serve -a 1
```

**Combined flags (for running "all" tests):**
```bash
flm serve -e 1 -a 1
```


### 5. Run Tests

Run tests with:

```bash
# Run all tests
flm-test --all

# Run specific tests
flm-test --llm                    # LLM tests only
flm-test --embedding              # Embedding tests only
flm-test --audio                  # Audio tests only
flm-test --vision                 # Vision tests only

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
  1. Initial prompt:"Tell me a joke and explain why it's funny." 
  2. Follow-up: "Summarize the joke and its explanation."

**Output:** `llm_results_v{version}_{timestamp}.csv`

### Vision Tests
Tests Vision-Language Models (VLM) with multi-image analysis.

**What it tests:**
- Multi-image understanding
- Detailed description generation
- Creative story generation connecting multiple images
- Streaming responses for image-to-text

**Test Images:**
- `test_files/image/test_image1.jpeg`
- `test_files/image/test_image2.jpg`

**Output:** `vison_results_v{version}_{timestamp}.csv`

## Understanding Results

Test results are saved as CSV files with the format:
```
{test_type}_results_v{version}_{timestamp}.csv
```

**Example filenames:**
- `llm_results_v0.9.35_20260308_202906.csv`
- `vision_results_vunknown_version_20260308_203124.csv`

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

### Interpreting Results

- **N/A**: Feature not supported by the model
- **ERROR: {message}**: Test failed with specific error
- **Empty content**: Model timeout or connection issue


