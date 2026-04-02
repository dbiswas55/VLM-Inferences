
"""Simple entry point for running a single multimodal inference request."""

from time import time
from pathlib import Path

from config import Config
from backends import get_backend_from_config
from backends.request import TextBlock, ImageBlock, InferenceRequest




# Available hostings: ollama, mlx_vlm, vllm, transformers, gemini, openai
# Available models: gemma3-4b, gemma3-12b, qwen3vl-4b, qwen3vl-8b, gemini-flash-2.5lite, gemini-flash-2.5
CLIENT_NAME = "ollama/gemma3-4b"  # Format: "hosting/model". Leave empty to use config 'active'.
DEBUG = True

SEPARATOR = "=" * 72
CONFIG_PATH = "configs/experiment.json"
IMAGE_PATHS = [
    "input/images/slide_020.png",
    "input/images/slide_021.png",
]

SYSTEM_PROMPT = ""
USER_PROMPT = "Describe the two images and then summarize the main information shown."


def get_client(cfg: Config, debug: bool = False) -> dict:
    client = cfg.get_client_by_name(CLIENT_NAME) if CLIENT_NAME else cfg.get_current_client()
    if debug:
        print(f"\n{SEPARATOR}")
        print(f"Hosting : {client['hosting']}")
        print(f"Client  : {client['name']}")
        print(f"Backend : {client['backend']}")
        print(f"Model   : {client['model_id']}")
        common_params = [
            f"max_tokens={client['max_tokens']}",
            f"temp={client['temperature']}",
            f"top_p={client['top_p']}",
        ]
        backend_params: list[str] = []

        if client["backend"] == "openai":
            print(f"Base URL: {client.get('base_url', 'https://api.openai.com/v1')}")
        elif client["backend"] == "gemini":
            backend_params.append(f"thinking_budget={client.get('thinking_budget')}")
        elif client["backend"] == "transformers":
            backend_params.append(f"quantization_level={client.get('quantization_level')}")

        all_params = common_params + backend_params
        print(f"Params  : {'  '.join(all_params)}")
        print(SEPARATOR)
    return client


def build_request_payload(
    system_prompt: str,
    prompt: str,
    image_paths: list[str],
    debug: bool = False,
) -> dict:
    if debug:
        print(f"Prompt : {prompt}")
        print(f"Images : {', '.join(image_paths)}")

    content = []
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        content.append(ImageBlock(str(path)))

    content.append(TextBlock(prompt))
    return {"content": content, "system_prompt": system_prompt}


def run_client(client: dict, payload: dict):
    request = InferenceRequest(
        content=payload["content"],
        system_prompt=payload["system_prompt"],
        max_new_tokens=client["max_tokens"],
        temperature=client["temperature"],
        top_p=client["top_p"],
    )

    backend = get_backend_from_config(client)

    start_time = time()
    answer = backend.run(request)
    elapsed = time() - start_time

    print(f"Status : OK ({elapsed:.2f}s)")
    print("Model output shown below")
    print(f"{SEPARATOR}")
    print(answer.strip())


def main(debug: bool = DEBUG):
    cfg = Config(CONFIG_PATH)
    client = get_client(cfg, debug=debug)
    payload = build_request_payload(SYSTEM_PROMPT, USER_PROMPT, IMAGE_PATHS, debug=debug)

    try:
        run_client(client, payload)
    except Exception as exc:
        print(f"\n{SEPARATOR}")
        print(f"Hosting : {client['hosting']}")
        print(f"Client : {client['name']}")
        print("Status : FAILED")
        print(f"Error  : {exc}")


if __name__ == "__main__":
    main()
