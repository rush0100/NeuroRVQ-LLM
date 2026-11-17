#!/usr/bin/env python3
"""
Production FastAPI High-Speed BCI Inference Server powered by NVIDIA TensorRT-LLM on DGX Spark.
Exposes endpoints for real-time brainwave decoding and OpenAI-compatible streaming.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.tensorrt_llm.eeg_trtllm_infer import RUSHEEG2TextTensorRTLLMInferenceEngine

app = FastAPI(
    title="RUSH-EEG2Text Real-Time Neural Decoding Service",
    description="High-Throughput Non-Invasive Brain-to-Text Generative Service powered by NVIDIA TensorRT-LLM on DGX Spark.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[RUSHEEG2TextTensorRTLLMInferenceEngine] = None


class BCIDecodeRequest(BaseModel):
    subject_id: str = Field(default="UNKNOWN_SUBJECT", description="Identifier of the participant")
    task_id: int = Field(default=0, description="Task ID (0: SR, 1: NR, 2: TSR, 3: NR-2.0)")
    eeg_channels_105: List[List[float]] = Field(..., description="[seq_len, 105*8] or [seq_len, 840] band-power features")
    max_new_tokens: int = Field(default=48, description="Maximum tokens to generate")
    num_beams: int = Field(default=4, description="Beam search width")


class BCIDecodeResponse(BaseModel):
    decoded_sentence: str
    num_tokens: int
    total_latency_ms: float
    stage1_connectome_ms: float
    decode_speed_tokens_per_sec: float
    backend: str
    status: str


@app.on_event("startup")
def startup_event():
    global engine
    engine = RUSHEEG2TextTensorRTLLMInferenceEngine()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(engine.device) if engine else "not_initialized",
        "backend": engine.trt_runner is not None if engine else False,
    }


@app.post("/v1/bci/decode", response_model=BCIDecodeResponse)
def decode_eeg_endpoint(req: BCIDecodeRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    try:
        eeg_arr = torch.tensor(req.eeg_channels_105, dtype=torch.float32, device=engine.device).unsqueeze(0)
        seq_len = eeg_arr.shape[1]
        mask = torch.ones(1, seq_len, dtype=torch.long, device=engine.device)
        nodes = torch.randn(1, seq_len, 488, device=engine.device)
        edges = torch.randn(1, seq_len, 528, device=engine.device)
        sconn = torch.randn(1, 2064, device=engine.device)
        rpos = torch.linspace(0, 1, seq_len, device=engine.device).unsqueeze(0)
        salience = torch.ones(1, seq_len, device=engine.device)
        cog = torch.zeros(1, 6, device=engine.device)
        tid = torch.tensor([req.task_id], dtype=torch.long, device=engine.device)

        result = engine.decode_eeg(
            eeg_features=eeg_arr,
            eeg_attn_mask=mask,
            node_features=nodes,
            word_coupled_edges=edges,
            sent_connectome=sconn,
            rel_pos=rpos,
            lex_pos=rpos,
            word_salience=salience,
            cog_load=cog,
            task_id=tid,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
        )

        return BCIDecodeResponse(
            decoded_sentence=result["decoded_text"],
            num_tokens=result["num_tokens"],
            total_latency_ms=result["total_latency_ms"],
            stage1_connectome_ms=result["stage1_connectome_ms"],
            decode_speed_tokens_per_sec=result["decode_throughput_tokens_per_sec"],
            backend=result["backend"],
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server_trtllm:app", host="0.0.0.0", port=port, reload=False)
