#!/usr/bin/env bash
# ============================================================
# Start the vLLM server with BioMistral-7B AWQ (4-bit)
# Must be run from the bio-med-rag directory.
# ============================================================

MODEL="${VLLM_MODEL:-BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM}"
PORT="${VLLM_PORT:-8080}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.7}"
MAX_LEN="${VLLM_MAX_LEN:-4096}"

echo "Starting vLLM server..."
echo "  Model       : $MODEL"
echo "  Quantization: awq"
echo "  Port        : $PORT"
echo "  GPU util    : $GPU_UTIL"
echo ""

VLLM_PLUGINS="" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization awq \
    --dtype half \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT"
