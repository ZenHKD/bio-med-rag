#!/usr/bin/env bash
# ============================================================
# Start the vLLM embedding server with PubMedBERT
# Must be run from the bio-med-rag directory.
# ============================================================

MODEL="${EMBED_MODEL:-NeuML/pubmedbert-base-embeddings}"
PORT="${EMBED_PORT:-8081}"
GPU_UTIL="${EMBED_GPU_UTIL:-0.15}"
HOST="${EMBED_HOST:-0.0.0.0}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/vllm_embed.log"

mkdir -p "$LOG_DIR"

echo "Starting vLLM embedding server..."
echo "  Model    : $MODEL"
echo "  Host     : $HOST"
echo "  Port     : $PORT"
echo "  GPU util : $GPU_UTIL"
echo "  Log      : $LOG_FILE"
echo ""

VLLM_PLUGINS="" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --runner pooling \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    2>&1 | tee -a "$LOG_FILE"
