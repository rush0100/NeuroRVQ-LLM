"""
Stage 2 Architecture: RUSH-EEG2Text (Discrete Brain Prefix -> Decoder-Only LLM with LoRA/QLoRA).
Supports BF16 LoRA (7B/8B) and 4-bit NF4 QLoRA (32B/70B) on NVIDIA DGX Spark 128 GB Unified Memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.models.cet_rvq_pretrainer import CETRVQPretrainer


class BrainToLLMProjector(nn.Module):
    """2-layer MLP adapter mapping quantized RVQ brain tokens [B, K, D_eeg] -> LLM embedding space [B, K, D_llm]."""

    def __init__(self, eeg_embed_dim: int, llm_hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(eeg_embed_dim),
            nn.Linear(eeg_embed_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
        )

    def forward(self, quantized_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(quantized_tokens)


def load_llm_and_tokenizer(
    model_name_or_path: str,
    quantization: str = "none",
    precision: str = "bf16",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    device_map: Optional[str | Dict[str, Any]] = None,
) -> Tuple[nn.Module, Any]:
    """Load causal LLM or Seq2Seq BART/T5 optimized for DGX Spark (BF16, full FT, or LoRA/QLoRA)."""
    hf_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    is_seq2seq = bool(getattr(hf_cfg, "is_encoder_decoder", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = (
        torch.bfloat16
        if precision == "bf16" and torch.cuda.is_available()
        else (torch.float16 if precision == "fp16" else torch.float32)
    )

    bnb_config = None
    if quantization == "nf4" and not is_seq2seq:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = device_map or "auto"
    elif device_map is not None:
        load_kwargs["device_map"] = device_map

    if is_seq2seq:
        llm = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **load_kwargs)
    else:
        llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    llm.config.use_cache = False

    if quantization == "nf4" and not is_seq2seq:
        llm = prepare_model_for_kbit_training(llm)

    if use_lora:
        default_targets = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"] if is_seq2seq else ["q_proj", "v_proj"]
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or default_targets,
            bias="none",
        )
        llm = get_peft_model(llm, peft_cfg)
    elif getattr(hf_cfg, "is_encoder_decoder", False):
        # For controlled BART-large baseline when use_lora=False, allow standard full seq2seq fine-tuning (matching CET-MAE / LURS!)
        for p in llm.parameters():
            p.requires_grad = True
    else:
        for p in llm.parameters():
            p.requires_grad = False

    return llm, tokenizer


class RUSHEEG2TextForCausalDecoding(nn.Module):
    """
    End-to-End Brain-to-LLM Generative Decoder (RUSH-EEG2Text):
      - Frozen/Fine-tunable Stage-1 CET-RVQ discrete neural tokenizer
      - Trainable BrainToLLMProjector
      - LoRA / QLoRA Decoder-only LLM
      - Scheduled target-token dropout to mitigate teacher-forcing exposure bias
      - Strict Teacher-Forcing-Free `.generate()` inference + Gaussian Noise Sanity Mode
    """

    def __init__(
        self,
        cet_rvq: CETRVQPretrainer,
        llm: nn.Module,
        tokenizer: Any,
        prompt_template: str = "Decode the following neural brain signal tokens into an exact English sentence:\n<eeg_prefix>\nDecoded sentence:",
        freeze_cet_rvq: bool = True,
    ) -> None:
        super().__init__()
        self.cet_rvq = cet_rvq
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

        if freeze_cet_rvq:
            for p in self.cet_rvq.parameters():
                p.requires_grad = False
        else:
            for p in self.cet_rvq.parameters():
                p.requires_grad = False
            # Only unfreeze the active encoder + RVQ representation path during Stage 2 causal LM tuning;
            # keep all Stage-1-only MAE decoders, reconstruction heads, contrastive heads, & aux heads frozen!
            for p in self.cet_rvq.eeg_encoder.parameters():
                p.requires_grad = True
            for p in self.cet_rvq.rvq.parameters():
                p.requires_grad = True

        self.is_seq2seq = bool(getattr(self.llm.config, "is_encoder_decoder", False))
        llm_hidden_dim = int(getattr(self.llm.config, "d_model", getattr(self.llm.config, "hidden_size", 1024)))
        self.projector = BrainToLLMProjector(
            eeg_embed_dim=self.cet_rvq.embed_dim,
            llm_hidden_dim=llm_hidden_dim,
        )

        parts = self.prompt_template.split("<eeg_prefix>")
        if len(parts) != 2:
            raise ValueError("prompt_template must contain exactly one `<eeg_prefix>` placeholder.")
        self.prompt_pre_str, self.prompt_post_str = parts

    def _get_input_embeddings_layer(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def _build_prefix_embeds(
        self,
        eeg_features: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        word_coupled_edges: Optional[torch.Tensor] = None,
        sent_connectome: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
        cog_load: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        replace_with_gaussian_noise: bool = False,
    ) -> torch.Tensor:
        """Convert Dual-Stream EEG batch (or pure Gaussian noise for sanity check) into LLM-space prefix embeddings [B, P, D_llm]."""
        if replace_with_gaussian_noise:
            bsz, max_l = eeg_attn_mask.shape
            rand_mask = torch.zeros_like(eeg_attn_mask)
            rand_rpos = torch.zeros_like(eeg_attn_mask, dtype=torch.float32)
            for bi in range(bsz):
                rlen = int(torch.randint(6, max(8, max_l // 2), (1,)).item())
                rand_mask[bi, :rlen] = 1
                rand_rpos[bi, :rlen] = torch.linspace(0.0, 1.0, steps=rlen, device=eeg_attn_mask.device)
            eeg_attn_mask = rand_mask
            rel_pos = rand_rpos
            lex_pos = rand_rpos
            eeg_features = torch.randn_like(eeg_features) * eeg_attn_mask.unsqueeze(-1).to(eeg_features.dtype)
            if node_features is not None:
                node_features = torch.randn_like(node_features) * eeg_attn_mask.unsqueeze(-1).to(node_features.dtype)
            if word_coupled_edges is not None:
                word_coupled_edges = torch.randn_like(word_coupled_edges) * eeg_attn_mask.unsqueeze(-1).to(word_coupled_edges.dtype)
            if sent_connectome is not None:
                sent_connectome = torch.randn_like(sent_connectome)
            if cog_load is not None:
                cog_load = torch.randn_like(cog_load)
            if word_salience is not None:
                word_salience = eeg_attn_mask.to(word_salience.dtype)

        with torch.set_grad_enabled(any(p.requires_grad for p in self.cet_rvq.parameters())):
            resampled = self.cet_rvq.eeg_encoder(
                eeg_features=eeg_features,
                eeg_attn_mask=eeg_attn_mask,
                node_features=node_features,
                word_coupled_edges=word_coupled_edges,
                sent_connectome=sent_connectome,
                rel_pos=rel_pos,
                lex_pos=lex_pos,
                word_salience=word_salience,
                cog_load=cog_load,
                task_id=task_id,
                apply_augment=self.training,
            )
            quantized = self.cet_rvq.rvq(resampled)["quantized"]

        emb_layer = self._get_input_embeddings_layer()
        target_device = emb_layer.weight.device
        target_dtype = emb_layer.weight.dtype

        brain_embeds = self.projector(quantized).to(device=target_device, dtype=target_dtype)  # [B, K, D_llm]
        if self.is_seq2seq:
            # For BART/T5 Encoder-Decoder, feed the K projected neural tokens directly into the bidirectional Encoder!
            return brain_embeds

        bsz = brain_embeds.size(0)
        pre_ids = self.tokenizer(self.prompt_pre_str, add_special_tokens=True, return_tensors="pt")["input_ids"].to(target_device)
        post_ids = self.tokenizer(self.prompt_post_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(target_device)

        pre_embeds = emb_layer(pre_ids).expand(bsz, -1, -1)
        post_embeds = emb_layer(post_ids).expand(bsz, -1, -1)
        return torch.cat([pre_embeds, brain_embeds, post_embeds], dim=1)

    def forward(
        self,
        eeg_features: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
        target_texts: List[str],
        node_features: Optional[torch.Tensor] = None,
        word_coupled_edges: Optional[torch.Tensor] = None,
        sent_connectome: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
        cog_load: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        max_text_len: int = 64,
        word_dropout_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with scheduled target-token dropout to prevent LLM over-dominance."""
        prefix_embeds = self._build_prefix_embeds(
            eeg_features=eeg_features,
            eeg_attn_mask=eeg_attn_mask,
            node_features=node_features,
            word_coupled_edges=word_coupled_edges,
            sent_connectome=sent_connectome,
            rel_pos=rel_pos,
            lex_pos=lex_pos,
            word_salience=word_salience,
            cog_load=cog_load,
            task_id=task_id,
            replace_with_gaussian_noise=False,
        )
        bsz, prefix_len, _ = prefix_embeds.shape
        target_device = prefix_embeds.device
        emb_layer = self._get_input_embeddings_layer()

        if self.is_seq2seq:
            tok = self.tokenizer(
                [t.strip() for t in target_texts],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_text_len,
                return_tensors="pt",
            )
            target_ids = tok["input_ids"].to(target_device)
            target_mask = tok["attention_mask"].to(target_device)
            target_labels = target_ids.clone()
            target_labels[target_mask == 0] = -100

            base_seq2seq = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
            dec_input_ids = base_seq2seq.prepare_decoder_input_ids_from_labels(labels=target_labels)
            if self.training and word_dropout_prob > 0.0:
                drop_mask = (torch.rand_like(dec_input_ids.float()) < word_dropout_prob) & (target_mask == 1)
                drop_mask[:, 0] = False  # preserve decoder_start_token_id
                dec_input_ids = dec_input_ids.clone()
                dec_input_ids[drop_mask] = self.tokenizer.pad_token_id

            enc_mask = torch.ones((bsz, prefix_len), dtype=target_mask.dtype, device=target_device)
            outputs = self.llm(
                inputs_embeds=prefix_embeds,
                attention_mask=enc_mask,
                decoder_input_ids=dec_input_ids,
                labels=target_labels,
            )
            return {"loss": outputs.loss}

        eos = self.tokenizer.eos_token or ""
        texts_with_eos = [f" {t.strip()}{eos}" for t in target_texts]
        tok = self.tokenizer(
            texts_with_eos,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        target_ids = tok["input_ids"].to(target_device)
        target_mask = tok["attention_mask"].to(target_device)

        # Scheduled exposure-bias mitigation: randomly replace input target tokens with pad/unk embedding
        # while still requiring the model to predict the true next token from the brain prefix!
        corrupted_input_ids = target_ids.clone()
        if self.training and word_dropout_prob > 0.0:
            drop_mask = (torch.rand_like(corrupted_input_ids.float()) < word_dropout_prob) & (target_mask == 1)
            corrupted_input_ids[drop_mask] = self.tokenizer.pad_token_id

        target_embeds = emb_layer(corrupted_input_ids)
        full_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

        prefix_mask = torch.ones((bsz, prefix_len), dtype=target_mask.dtype, device=target_device)
        full_attn_mask = torch.cat([prefix_mask, target_mask], dim=1)

        ignore_prefix = torch.full((bsz, prefix_len), -100, dtype=torch.long, device=target_device)
        target_labels = target_ids.clone()
        target_labels[target_mask == 0] = -100
        full_labels = torch.cat([ignore_prefix, target_labels], dim=1)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_mask,
            labels=full_labels,
        )
        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate_teacher_forcing_free(
        self,
        eeg_features: torch.Tensor,
        eeg_attn_mask: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        word_coupled_edges: Optional[torch.Tensor] = None,
        sent_connectome: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
        lex_pos: Optional[torch.Tensor] = None,
        word_salience: Optional[torch.Tensor] = None,
        cog_load: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        max_new_tokens: int = 48,
        num_beams: int = 4,
        repetition_penalty: float = 1.25,
        no_repeat_ngram_size: int = 3,
        replace_with_gaussian_noise: bool = False,
    ) -> List[str]:
        """
        Strict Teacher-Forcing-Free Open-Vocabulary Generation.
        No ground-truth text tokens are ever passed during inference.
        """
        self.eval()
        prefix_embeds = self._build_prefix_embeds(
            eeg_features=eeg_features,
            eeg_attn_mask=eeg_attn_mask,
            node_features=node_features,
            word_coupled_edges=word_coupled_edges,
            sent_connectome=sent_connectome,
            rel_pos=rel_pos,
            lex_pos=lex_pos,
            word_salience=word_salience,
            cog_load=cog_load,
            task_id=task_id,
            replace_with_gaussian_noise=replace_with_gaussian_noise,
        )
        bsz, prefix_len, _ = prefix_embeds.shape
        attn_mask = torch.ones((bsz, prefix_len), dtype=torch.long, device=prefix_embeds.device)

        prev_cache = bool(getattr(self.llm.config, "use_cache", False))
        self.llm.config.use_cache = True
        try:
            gen_ids = self.llm.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                use_cache=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            self.llm.config.use_cache = prev_cache

        # HuggingFace causal LM `.generate(inputs_embeds=...)` returns only newly generated token IDs
        # (or prefixed if model-specific); strip any prompt artifacts cleanly.
        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [d.strip() for d in decoded]


