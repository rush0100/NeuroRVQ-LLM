# ==============================================================================
# Dockerfile: High-Performance NVIDIA TensorRT-LLM Serving for RUSH-EEG2Text
# Optimized for NVIDIA Grace Blackwell (GB10 Superchip) and Hopper GPUs
# ==============================================================================

FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /workspace

# Install high-throughput inference dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0 \
    pydantic==2.10.4 \
    transformers==4.49.0 \
    peft==0.14.0 \
    accelerate==1.4.0 \
    sentencepiece==0.2.0 \
    protobuf==5.29.3 \
    tiktoken==0.8.0

# Install TensorRT-LLM Python wheel if available for Blackwell/CUDA 12.9
RUN pip install --no-cache-dir tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com || true

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "deploy.tensorrt_llm.server_trtllm:app", "--host", "0.0.0.0", "--port", "8000"]
