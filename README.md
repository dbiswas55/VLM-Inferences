# VLM-Inferences

A lightweight, unified framework for **Vision-Language Model (VLM) inference** that lets you switch between local and cloud-hosted models with a single config change. Run multimodal prompts — interleaved text and images — against Ollama, MLX-VLM, vLLM, HuggingFace Transformers, Gemini, OpenAI, or Anthropic without rewriting any inference code.

## Key Features

- **Unified inference interface** — one `InferenceRequest` with `TextBlock` and `ImageBlock` works across every backend.
- **Multiple backends, one config** — swap between 7 hosting providers (local and cloud) by editing a JSON file.
- **Interleaved multimodal content** — freely mix text segments and local images in any order within a single request.
- **Structured JSON configuration** — all models, parameters, datasets, and prompt workflows live in a single, readable config file.
- **Extensible workflow system** — define multi-step prompt workflows (e.g. chain-of-thought) in config with external prompt templates.
- **Lazy-loaded clients** — backend SDKs and models are only loaded when first used, keeping startup fast.
- **HuggingFace model management** — built-in helpers to download, list, and delete cached models.

## Supported Backends

| Category | Backend | Hosting Key | How it runs |
|---|---|---|---|
| **Cloud API** | Google Gemini (native SDK) | `gemini` | API call via `google-genai` |
| **Cloud API** | Google Gemini (OpenAI-compat) | `gemini_compat` | OpenAI-compatible endpoint |
| **Cloud API** | OpenAI | `openai` | GPT-4o, GPT-4o-mini |
| **Cloud API** | Anthropic | `anthropic` | Claude via OpenAI-compatible endpoint |
| **Local Server** | Ollama | `ollama` | Local server on port 11434 |
| **Local Server** | MLX-VLM | `mlx_vlm` | Apple Silicon, port 8080 |
| **Local Server** | vLLM | `vllm` | CUDA GPU, port 8000 |
| **In-Process** | HuggingFace Transformers | `transformers` | Direct model loading (CUDA / MPS / CPU) |

Pre-configured models include **Gemma 3** (4B, 12B) and **Qwen3-VL** (4B, 8B) across all local backends, plus Gemini and GPT-4o for cloud.

## Project Structure

```
VLM-Inferences/
├── configs/
│   └── experiment.json          # All model, dataset, and workflow configuration
├── input/
│   └── images/                  # Input images for inference
├── output/                      # Per-item workflow results (created on run)
├── src/
│   ├── test_backends.py         # Run a single multimodal request (backend smoke test)
│   ├── test_workflows.py        # Run a single-/multi-step workflow over a dataset
│   ├── backends/
│   │   ├── __init__.py          # Backend factory (get_backend_from_config)
│   │   ├── backends.py          # BaseBackend, GeminiBackend, OpenAIBackend, TransformersBackend
│   │   └── request.py           # TextBlock, ImageBlock, InferenceRequest
│   ├── utils/
│   │   └── config.py            # Config loader with structured accessors
│   ├── prepare/
│   │   └── prepare_backends.py  # Backend setup guide + HuggingFace model management
│   └── prompts/                 # Prompt template files (referenced by workflows)
├── README.md
└── .env                         # API keys and HF token (not committed)
```

## Quick Start

### 1. Create Environment

```bash
python3 -m venv venv312
source venv312/bin/activate      # macOS / Linux
# venv312\Scripts\activate       # Windows
```

### 2. Install Dependencies

```bash
pip install mlx mlx-vlm torch torchvision Pillow transformers accelerate \
            huggingface_hub python-dotenv openai google-genai
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_token_here       # huggingface.co/settings/tokens
HF_HOME=.cache/huggingface         # optional custom cache path
GEMINI_API_KEY=...                 # for Gemini backend
OPENAI_API_KEY=...                 # for OpenAI backend
ANTHROPIC_API_KEY=...              # for Anthropic backend
```

### 4. Set Up a Local Backend (Optional)

**Ollama** (easiest to start with):

```bash
brew install ollama                # macOS
ollama pull gemma3:4b
ollama serve                       # http://localhost:11434/v1
```

**MLX-VLM** (Apple Silicon):

```bash
python -m mlx_vlm.server --model mlx-community/gemma-3-4b-it-qat-4bit --port 8080
```

**vLLM** (CUDA):

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

See [`src/prepare/prepare_backends.py`](src/prepare/prepare_backends.py) for the full setup guide and HuggingFace model download utilities.

### 5. Run Inference

Two entry points are provided. Both select a client via `CLIENT_NAME = "hosting/model"`
(leave empty to use whichever client is set as `active` in the config).

**Single request** — the quickest way to confirm a backend is wired up. Edit the
constants at the top of [`src/test_backends.py`](src/test_backends.py):

```python
CLIENT_NAME = "ollama/gemma3-4b"   # Format: "hosting/model"
IMAGE_PATHS = ["input/images/slide_020.png", "input/images/slide_021.png"]
USER_PROMPT = "Describe the two images and then summarize the main information shown."
```

```bash
python src/test_backends.py
```

**Workflow over a dataset** — run a single- or multi-step prompt workflow (see
[Workflows](#workflows)) over every item in a configured dataset:

```bash
python src/test_workflows.py                                   # defaults from the file
python src/test_workflows.py --workflow onestep_summary --client gemini/flash-2.5
python src/test_workflows.py --dataset demo_images --debug
```

If the dataset defines an `output_dir`, each item's results are saved to
`<output_dir>/<workflow>/<item_id>.json`.

## Configuration

All settings live in [`configs/experiment.json`](configs/experiment.json). The structure:

```jsonc
{
  "models": {
    "active": { "hosting": "ollama", "model": "gemma3-4b" },  // default client
    "defaults": { "max_tokens": 4096, "temperature": 0.3, "top_p": 1.0 },
    "hostings": {
      "ollama": {
        "backend": "openai",
        "base_url": "http://localhost:11434/v1",
        "models": [
          { "name": "gemma3-4b", "model_id": "gemma3:4b" },
          // ...
        ]
      },
      // gemini, openai, anthropic, mlx_vlm, vllm, transformers ...
    }
  },
  "processing": { "batch_size": 1, "output_format": "jsonl" },
  "datasets": {
    "demo_images": {
      "name": "demo_images",
      "root_dir": "input/images",     // images to process, relative to project root
      "output_dir": "output/demo_images"  // where results are saved (omit to skip saving)
    }
  },
  "prompts": {
    "prompt_root": "src/prompts",
    "workflows": {
      "onestep_summary": { "steps": [{ "system": "", "user": "summary/v1_prompt.txt" }] },
      "multisteps_summary": { "steps": [
          { "system": "", "user": "summary/v2_step1.txt" },
          { "system": "", "user": "summary/v2_step2.txt" }
        ]
      }
    }
  }
}
```

**Selecting a client** — either set `models.active` in the config, or specify `CLIENT_NAME = "hosting/model"` in code. Model-level fields override hosting-level fields, which override `defaults`.

### Workflows

The `prompts.workflows` section defines reusable single- or multi-step prompt
pipelines. Each step references a system and user prompt (inline string or path
to a `.txt` file under `prompt_root`). `src/test_workflows.py` runs a workflow
over every item in a dataset, feeding each step's output into the next.

Each step's user template can reference two kinds of tags, filled only when the
tag appears in that step:

- **item tag** (e.g. `{images}`) — filled from the work item itself (its inputs).
- **parsed tag** (e.g. `{response_text}`) — filled from the previous step's parsed
  result. The default parser exposes its output as `{response_text}`; a step can
  register a custom parser to expose additional or differently-named tags (the
  shipped `multisteps_summary` uses this to pass `{image_summaries}` from step 1
  into step 2). A parsed tag resolves only if that exact key comes back from the
  parser, so keep parser output keys and template tags in sync.

Two example workflows ship with the template:

| Workflow | Steps | What it does |
|---|---|---|
| `onestep_summary` | 1 | All images sent together; one combined summary returned. |
| `multisteps_summary` | 2 | Step 1 describes the images (parsed as `{image_summaries}`); step 2 synthesises them into one summary. |

When a dataset defines `output_dir`, each item's results — every step's parsed
tags (intermediate outputs plus the final `response_text`) — are saved to
`<output_dir>/<workflow>/<item_id>.json`.

## Inference Request Format

The core abstraction is `InferenceRequest` — an ordered list of `TextBlock` and `ImageBlock` items that every backend understands:

```python
from backends.request import TextBlock, ImageBlock, InferenceRequest

request = InferenceRequest(
    content=[
        ImageBlock("input/images/slide_1.png"),
        TextBlock("What does this diagram show?"),
        ImageBlock("input/images/slide_2.png"),
        TextBlock("How does this compare to the previous slide?"),
    ],
    system_prompt="You are a helpful assistant.",
    max_new_tokens=4096,
    temperature=0.3,
    top_p=1.0,
)
```

Images are automatically encoded (base64 data URI for OpenAI-compatible backends, raw bytes for Gemini, PIL for Transformers). You compose the content sequence however you like — the backend handles the rest.

## HuggingFace Model Management

The prepare script doubles as a model manager:

```bash
python src/prepare/prepare_backends.py
```

Available functions:

| Function | Description |
|---|---|
| `download_model(model_id)` | Download a model to the HF cache |
| `list_cached_models()` | List all cached models with sizes |
| `delete_cached_model(model_id)` | Delete a specific model |
| `delete_cached_model_interactive()` | Interactive picker to delete models |

## Adding a New Backend

1. Add a hosting entry in `configs/experiment.json` under `models.hostings`.
2. If the service speaks the OpenAI chat completions API, set `"backend": "openai"` — no code changes needed.
3. For a custom protocol, subclass `BaseBackend` in `src/backends/backends.py`, implement `run(request) -> str`, and register it in `src/backends/__init__.py`.

## License

This project is open source. See [LICENSE](LICENSE) for details.
