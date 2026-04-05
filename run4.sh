#!/bin/bash
#SBATCH -J tfBench
#SBATCH -o tf_bench.o%j
#SBATCH --mail-user=dipayan1109033@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node=1 -N 1
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=ada:2

set -euo pipefail

cd /project/subhlok/dipayan/VLM-Inferences
source /project/subhlok/dipayan/my_envs/venv312/bin/activate
set -a; source .env; set +a

JOB_CACHE="/project/subhlok/dipayan/caches/job_${SLURM_JOB_ID}"
export HOME="$JOB_CACHE/home"
export TMPDIR="$JOB_CACHE/tmp"
export TRITON_CACHE_DIR="$JOB_CACHE/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor-cache"

mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
env | grep -E 'HOME=|SCRATCH|TMP|TRITON|TORCHINDUCTOR|XDG|VLLM|HF_HOME' | sort

python tests/benchmark_transformers_only.py