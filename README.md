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
├── src/
│   ├── inference.py             # Main entry point — run a multimodal inference
│   ├── config.py                # Config loader with structured accessors
│   ├── backends/
│   │   ├── __init__.py          # Backend factory (get_backend_from_config)
│   │   ├── backends.py          # BaseBackend, GeminiBackend, OpenAIBackend, TransformersBackend
│   │   └── request.py           # TextBlock, ImageBlock, InferenceRequest
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

Edit the top of `src/inference.py` to select your client and prompt:

```python
CLIENT_NAME = "ollama/gemma3-4b"   # Format: "hosting/model"
IMAGE_PATHS = ["input/images/slide_020.png", "input/images/slide_021.png"]
USER_PROMPT = "Describe the two images and then summarize the main information shown."
```

Then run:

```bash
cd src
python inference.py
```

You can also leave `CLIENT_NAME` empty to use whichever client is set as `active` in the config.

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
  "datasets": { ... },
  "prompts": {
    "prompt_root": "src/prompts",
    "workflows": {
      "basic_captioning": { "steps": [{ "system": "", "user": "basic_captioning/step1_user.txt" }] },
      "chain_of_thought": { "steps": [/* multi-step workflow */] }
    }
  }
}
```

**Selecting a client** — either set `models.active` in the config, or specify `CLIENT_NAME = "hosting/model"` in code. Model-level fields override hosting-level fields, which override `defaults`.

### Workflows

The `prompts.workflows` section defines reusable multi-step prompt pipelines. Each step references a system and user prompt (inline string or path to a `.txt` file under `prompt_root`). This structure supports implementing different VLM workflows — basic captioning, chain-of-thought reasoning, or any custom pipeline you design.

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
