#!/usr/bin/env python3
"""
Step 1 for TensorRT-LLM: Merge LoRA weights into Qwen2.5-7B-Instruct base model
and export the unified HuggingFace model checkpoint ready for TensorRT-LLM engine build.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export and merge LoRA weights for TensorRT-LLM conversion")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default="./models/Qwen2.5-7B-Instruct",
        help="Path to base Qwen2.5-7B-Instruct weights",
    )
    parser.add_argument(
        "--lora_adapter_dir",
        type=str,
        default="./checkpoints/rush_eeg2text_full4task_champion_qwen7b/stage2_lora_adapter",
        help="Path to trained PEFT LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./deploy/models/qwen2.5-7b-eeg-merged",
        help="Path to save merged HuggingFace model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to perform weight merging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("RUSH-EEG2Text: Merging LoRA Weights into Base LLM for TensorRT-LLM")
    print(f"Base Model:    {args.base_model_dir}")
    print(f"LoRA Adapter:  {args.lora_adapter_dir}")
    print(f"Output Target: {args.output_dir}")
    print(f"Device:        {args.device}")
    print("=" * 80)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(out_path)

    # 2. Load Base Model in BF16
    print("[2/4] Loading base model in BF16 precision...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )

    # 3. Load & Merge LoRA Adapter
    if os.path.exists(args.lora_adapter_dir) and (
        os.path.exists(os.path.join(args.lora_adapter_dir, "adapter_model.bin"))
        or os.path.exists(os.path.join(args.lora_adapter_dir, "adapter_model.safetensors"))
    ):
        print("[3/4] Loading and merging LoRA weights...")
        lora_model = PeftModel.from_pretrained(base_model, args.lora_adapter_dir)
        merged_model = lora_model.merge_and_unload()
        print("  -> LoRA adapter successfully merged and unloaded!")
    else:
        print(f"[3/4] Notice: No adapter weights found in {args.lora_adapter_dir}. Using base model weights directly.")
        merged_model = base_model

    # 4. Save Merged Model & Tokenizer
    print(f"[4/4] Saving merged model to {out_path}...")
    merged_model.save_pretrained(out_path, safe_serialization=True)
    print("=" * 80)
    print(f"Merge Complete! Merged model saved at: {out_path.resolve()}")
    print("Ready for: bash deploy/tensorrt_llm/build_engine.sh")
    print("=" * 80)


if __name__ == "__main__":
    main()
