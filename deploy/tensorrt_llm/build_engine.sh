#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# TensorRT-LLM Engine Build Script for RUSH-EEG2Text (Qwen2.5-7B) on NVIDIA DGX Spark
# ==============================================================================
#
# Hardware Targets: NVIDIA Grace Blackwell (GB10 Superchip) & Hopper GPUs
# Features:
#   - Paged KV Cache with 64-token memory blocks
#   - In-Flight Batching (IFB)
#   - GPT Attention Plugin (FlashAttention-3 kernel on Blackwell)
#   - BFloat16 precision
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MERGED_MODEL_DIR="${PROJECT_DIR}/deploy/models/qwen2.5-7b-eeg-merged"
CKPT_DIR="${PROJECT_DIR}/deploy/models/qwen2.5-7b-trt_ckpt"
ENGINE_DIR="${PROJECT_DIR}/deploy/models/qwen2.5-7b-engine"

MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-16}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-256}
MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN:-128}
DTYPE=${DTYPE:-"bfloat16"}

echo "================================================================================"
echo "NVIDIA TensorRT-LLM Engine Compiler for RUSH-EEG2Text"
echo "Merged Model:   ${MERGED_MODEL_DIR}"
echo "TRT Checkpoint: ${CKPT_DIR}"
echo "Output Engine:  ${ENGINE_DIR}"
echo "Dtype:          ${DTYPE}"
echo "Max Batch Size: ${MAX_BATCH_SIZE}"
echo "Max Input Len:  ${MAX_INPUT_LEN}"
echo "Max Output Len: ${MAX_OUTPUT_LEN}"
echo "================================================================================"

mkdir -p "${CKPT_DIR}"
mkdir -p "${ENGINE_DIR}"

# Step 1: Convert Hugging Face Checkpoint to TensorRT-LLM intermediate format
echo "[Step 1/2] Converting Hugging Face model to TensorRT-LLM checkpoint..."
trtllm-convert-checkpoint \
    --model_dir "${MERGED_MODEL_DIR}" \
    --output_dir "${CKPT_DIR}" \
    --dtype "${DTYPE}" \
    --workers 4 \
    2>&1 || python3 -m tensorrt_llm.models.qwen.convert \
        --model_dir "${MERGED_MODEL_DIR}" \
        --output_dir "${CKPT_DIR}" \
        --dtype "${DTYPE}"

# Step 2: Build TensorRT-LLM High-Performance Engine
echo "[Step 2/2] Compiling TensorRT-LLM execution engine with Blackwell/Hopper plugins..."
trtllm-build \
    --checkpoint_dir "${CKPT_DIR}" \
    --output_dir "${ENGINE_DIR}" \
    --gemm_plugin auto \
    --gpt_attention_plugin auto \
    --paged_kv_cache enable \
    --tokens_per_block 64 \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_input_len "${MAX_INPUT_LEN}" \
    --max_output_len "${MAX_OUTPUT_LEN}" \
    --use_custom_all_reduce enable

echo "================================================================================"
echo "TensorRT-LLM Engine Build Succeeded!"
echo "Engine Path: ${ENGINE_DIR}"
echo "Ready for high-throughput inference: python3 deploy/tensorrt_llm/eeg_trtllm_infer.py"
echo "================================================================================"
