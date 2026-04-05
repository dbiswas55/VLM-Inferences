"""Repeated thread-pooled benchmark for the vLLM backend only.

Measures:
- backend creation time
- first completed request time
- later-request average
- total benchmark time
- optional server startup / time-to-first-answer if provided by env vars

Optional env vars from Slurm:
- VLLM_SERVER_LAUNCH_TS
- VLLM_SERVER_READY_TS
Both should be set using time.perf_counter()-style monotonic values from Python,
or omitted. If omitted, startup-related fields are reported as unavailable.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter

from config import Config
from backends import get_backend_from_config
from backends.request import TextBlock, ImageBlock, InferenceRequest


CLIENT_NAME = "vllm/gemma3-12b"
DEBUG = True

CONFIG_PATH = "configs/experiment.json"
IMAGE_PATHS = [
    "input/images/slide_020.png",
    "input/images/slide_021.png",
]

SYSTEM_PROMPT = ""
USER_PROMPT = "Summarize all the above slides and provide the main information shown."
NUM_ITERATIONS = 20
MAX_WORKERS = 10
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
        print(f"Base URL: {client.get('base_url', 'https://api.openai.com/v1')}")
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


def run_one(backend, request: InferenceRequest, idx: int, benchmark_zero: float) -> dict:
    start = perf_counter()
    answer = backend.run(request)
    end = perf_counter()
    return {
        "idx": idx,
        "latency_sec": end - start,
        "completed_at_sec": end - benchmark_zero,
        "answer": answer.strip(),
    }


def parse_optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main(debug: bool = DEBUG) -> None:
    overall_start = perf_counter()

    cfg = Config(CONFIG_PATH)
    client = get_client(cfg, debug=debug)
    payload = build_request_payload()

    backend_create_start = perf_counter()
    backend = get_backend_from_config(client)
    backend_create_elapsed = perf_counter() - backend_create_start

    print(f"Iterations  : {NUM_ITERATIONS}")
    print(f"Max workers : {MAX_WORKERS}")
    print(f"Backend creation time (client only): {backend_create_elapsed:.2f}s")

    results = []
    benchmark_start = perf_counter()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(NUM_ITERATIONS):
            request = build_request(client, payload)
            futures.append(executor.submit(run_one, backend, request, i, benchmark_start))

        for done_count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            print(
                f"[{done_count}/{NUM_ITERATIONS}] "
                f"iter={row['idx']} {row['latency_sec']:.2f}s "
                f"(completed at {row['completed_at_sec']:.2f}s)"
            )

    benchmark_elapsed = perf_counter() - benchmark_start
    overall_elapsed = perf_counter() - overall_start

    results.sort(key=lambda x: x["idx"])
    latencies = [r["latency_sec"] for r in results]
    avg_latency = mean(latencies)
    throughput = NUM_ITERATIONS / benchmark_elapsed if benchmark_elapsed > 0 else 0.0
    first_answer = results[0]["answer"] if results else ""

    first_completed = min(results, key=lambda x: x["completed_at_sec"])
    first_completion_time = first_completed["completed_at_sec"]
    first_completed_latency = first_completed["latency_sec"]

    later_latencies = [
        r["latency_sec"] for r in results
        if r["idx"] != first_completed["idx"]
    ]
    later_avg = mean(later_latencies) if later_latencies else 0.0
    estimated_extra_first_request_cost = max(0.0, first_completed_latency - later_avg)

    # Optional server-side timing if exported by Slurm
    server_launch_ts = parse_optional_float_env("VLLM_SERVER_LAUNCH_TS")
    server_ready_ts = parse_optional_float_env("VLLM_SERVER_READY_TS")

    server_startup_sec = None
    startup_plus_first_answer_sec = None

    # These only make sense if the env timestamps came from the same perf_counter clock source.
    if server_launch_ts is not None and server_ready_ts is not None:
        server_startup_sec = server_ready_ts - server_launch_ts
        # Python script starts after server ready, so total to first answer =
        # server startup + first completed request inside this script
        startup_plus_first_answer_sec = server_startup_sec + first_completion_time

    print(f"\n{SEPARATOR}")
    print("Benchmark : vllm-threadpool")
    print(f"Total benchmark time                 : {benchmark_elapsed:.2f}s")
    print(f"Total script wall time               : {overall_elapsed:.2f}s")
    print(f"Backend creation time                : {backend_create_elapsed:.2f}s")
    print(f"First completed request idx          : {first_completed['idx']}")
    print(f"First completed request latency      : {first_completed_latency:.2f}s")
    print(f"First completion time from start     : {first_completion_time:.2f}s")
    print(f"Later-request average latency        : {later_avg:.2f}s")
    print(f"Estimated extra first-request cost   : {estimated_extra_first_request_cost:.2f}s")
    print(f"Average latency (all requests)       : {avg_latency:.2f}s")
    print(f"Min latency                          : {min(latencies):.2f}s")
    print(f"Max latency                          : {max(latencies):.2f}s")
    print(f"Overall req/sec                      : {throughput:.3f}")

    if server_startup_sec is not None:
        print(f"Server startup time                  : {server_startup_sec:.2f}s")
        print(f"Startup + first answer               : {startup_plus_first_answer_sec:.2f}s")
    else:
        print("Server startup time                  : unavailable")
        print("Startup + first answer               : unavailable")

    print(SEPARATOR)
    print("Sample first output preview:")
    print(first_answer[:OUTPUT_PREVIEW_CHARS])
    print(SEPARATOR)


if __name__ == "__main__":
    main()