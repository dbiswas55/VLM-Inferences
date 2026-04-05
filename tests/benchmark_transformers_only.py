"""Repeated sequential benchmark for the Transformers backend only.

Reports:
- backend construction time
- first-call time
- later-call average
- estimated initialization time inside first call
- total wall time
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from time import perf_counter

from config import Config
from backends import get_backend_from_config
from backends.request import TextBlock, ImageBlock, InferenceRequest


CLIENT_NAME = "transformers/gemma3-12b"
DEBUG = True

CONFIG_PATH = "configs/experiment.json"
IMAGE_PATHS = [
    "input/images/slide_020.png",
    "input/images/slide_021.png",
]

SYSTEM_PROMPT = ""
USER_PROMPT = "Summarize all the above slides and provide the main information shown."
NUM_ITERATIONS = 20
OUTPUT_PREVIEW_CHARS = 500

SEPARATOR = "=" * 72


def get_client(cfg: Config, debug: bool = False) -> dict:
    client = cfg.get_client_by_name(CLIENT_NAME)
    if debug:
        print(f"\n{SEPARATOR}")
        print(f"Hosting : {client['hosting']}")
        print(f"Client  : {client['name']}")
        print(f"Backend : {client['backend']}")
        print(f"Model   : {client['model_id']}")
        print(
            f"Params  : max_tokens={client['max_tokens']}  "
            f"temp={client['temperature']}  top_p={client['top_p']}"
        )
        print(SEPARATOR)
    return client


def build_request_payload() -> dict:
    content = []
    for image_path in IMAGE_PATHS:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        content.append(ImageBlock(str(path)))
        content.append(TextBlock("Describe this slide briefly."))
    content.append(TextBlock(USER_PROMPT))
    return {"content": content, "system_prompt": SYSTEM_PROMPT}


def build_request(client: dict, payload: dict) -> InferenceRequest:
    return InferenceRequest(
        content=payload["content"],
        system_prompt=payload["system_prompt"],
        max_new_tokens=client["max_tokens"],
        temperature=client["temperature"],
        top_p=client["top_p"],
    )


def main(debug: bool = DEBUG) -> None:
    overall_start = perf_counter()

    cfg = Config(CONFIG_PATH)
    client = get_client(cfg, debug=debug)
    payload = build_request_payload()

    backend_create_start = perf_counter()
    backend = get_backend_from_config(client)
    backend_create_elapsed = perf_counter() - backend_create_start

    latencies: list[float] = []
    first_answer = ""

    print(f"Iterations : {NUM_ITERATIONS}")
    print(f"Backend creation time (without model load): {backend_create_elapsed:.2f}s")

    benchmark_start = perf_counter()

    for i in range(NUM_ITERATIONS):
        request = build_request(client, payload)
        call_start = perf_counter()
        answer = backend.run(request)
        call_elapsed = perf_counter() - call_start
        latencies.append(call_elapsed)

        if i == 0:
            first_answer = answer.strip()

        print(f"[{i+1}/{NUM_ITERATIONS}] {call_elapsed:.2f}s")

    benchmark_elapsed = perf_counter() - benchmark_start
    overall_elapsed = perf_counter() - overall_start

    first_call_time = latencies[0]
    later_latencies = latencies[1:]
    later_avg = mean(later_latencies) if later_latencies else 0.0
    estimated_init_inside_first_call = max(0.0, first_call_time - later_avg)

    avg_latency_all = mean(latencies)
    throughput_all = NUM_ITERATIONS / benchmark_elapsed if benchmark_elapsed > 0 else 0.0

    steady_state_throughput = (len(later_latencies) / sum(later_latencies)) if later_latencies else 0.0

    print(f"\n{SEPARATOR}")
    print("Benchmark : transformers-sequential")
    print(f"Total benchmark time             : {benchmark_elapsed:.2f}s")
    print(f"Total script wall time           : {overall_elapsed:.2f}s")
    print(f"Backend creation time            : {backend_create_elapsed:.2f}s")
    print(f"First call time                  : {first_call_time:.2f}s")
    print(f"Later-call average               : {later_avg:.2f}s")
    print(f"Estimated init inside first call : {estimated_init_inside_first_call:.2f}s")
    print(f"Average latency (all calls)      : {avg_latency_all:.2f}s")
    print(f"Min latency                      : {min(latencies):.2f}s")
    print(f"Max latency                      : {max(latencies):.2f}s")
    print(f"Overall req/sec                  : {throughput_all:.3f}")
    print(f"Steady-state req/sec             : {steady_state_throughput:.3f}")
    print(SEPARATOR)
    print("Sample first output preview:")
    print(first_answer[:OUTPUT_PREVIEW_CHARS])
    print(SEPARATOR)


if __name__ == "__main__":
    main()