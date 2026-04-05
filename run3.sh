#!/bin/bash
#SBATCH -J vlmInfer
#SBATCH -o vllm_infer.o%j
#SBATCH --mail-user=dipayan1109033@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=ada:1

set -euo pipefail

cd /project/subhlok/dipayan/VLM-Inferences
source /project/subhlok/dipayan/my_envs/venv312/bin/activate
set -a; source .env; set +a

# JOB_CACHE="/project/subhlok/dipayan/caches/vllm-cache-run"
# rm -rf "$JOB_CACHE"
# mkdir -p "$JOB_CACHE"

JOB_CACHE="/project/subhlok/dipayan/caches/job_${SLURM_JOB_ID}"
export HOME="$JOB_CACHE/home"
export TMPDIR="$JOB_CACHE/tmp"
export VLLM_CACHE_ROOT="$JOB_CACHE/vllm-cache"
export TRITON_CACHE_DIR="$JOB_CACHE/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor-cache"

mkdir -p "$HOME" "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
env | grep -E 'HOME=|SCRATCH|TMP|TRITON|TORCHINDUCTOR|XDG|VLLM|HF_HOME' | sort

MODEL="google/gemma-3-12b-it"
# MODEL="Qwen/Qwen3-VL-8B-Instruct"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.95 \
  --generation-config vllm &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM server did not start within 10 minutes."
        exit 1
    fi
    sleep 5
done

python src/inference.py