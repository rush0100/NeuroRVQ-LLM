import json
import os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ZuCoEEGTextDataset, collate_eeg_text, load_and_split_zuco
from src.models.cet_rvq_pretrainer import CETRVQPretrainer
from src.train_stage1 import load_or_build_text_cache

torch.set_num_threads(8)
cfg = yaml.safe_load(open("configs/q1_action3_bart_large_singletask.yaml"))
train_ds, val_ds, test_ds = load_and_split_zuco(
    dataset_root=cfg["data"]["dataset_root"],
    tasks=["task1-SR"],
    split_strategy="subject_independent",
    held_out_test_subjects=["ZMG"],
    held_out_val_subjects=["ZJS"],
)
unique_sents = sorted({s["target_text"] for s in (train_ds.samples + val_ds.samples + test_ds.samples)})
_, text_bank_matrix, sent_to_id, llm_dim = load_or_build_text_cache(
    unique_sentences=unique_sents,
    dataset_root=cfg["data"]["dataset_root"],
    llm_name_or_path="./models/Qwen2.5-7B-Instruct",
    device=torch.device("cpu"),
)

s1 = cfg["stage1_cet_rvq"]
model = CETRVQPretrainer(
    eeg_feature_dim=840,
    num_channels=105,
    num_bands=8,
    embed_dim=s1["embed_dim"],
    llm_hidden_dim=llm_dim,
    encoder_layers=s1["encoder_layers"],
    encoder_heads=s1["encoder_heads"],
    ffn_dim=s1["ffn_dim"],
    dropout=s1["dropout"],
    use_resampler=True,
    num_latent_queries=s1["num_latent_queries"],
    rvq_num_quantizers=s1["rvq_num_quantizers"],
    rvq_codebook_size=s1["rvq_codebook_size"],
    rvq_commitment_weight=s1["rvq_commitment_weight"],
    rvq_quantizer_dropout=s1["rvq_quantizer_dropout"],
    rvq_ema_decay=s1["rvq_ema_decay"],
    max_eeg_len=66,
    stream_mode=s1["stream_mode"],
    proj_dropout=s1["proj_dropout"],
    max_logit_scale=s1["max_logit_scale"],
).eval()
ckpt = torch.load(
    "checkpoints/q1_action3_neurorvq_bart_large_singletask/stage1_cet_rvq_best.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model_state_dict"])

with torch.no_grad():
    bank_proj = model.project_text_bank(text_bank_matrix)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_eeg_text)
    all_b = list(loader)

    def eval_retrieval(embs, gts):
        sims = embs @ bank_proj.t()
        ts = sims.gather(1, gts.unsqueeze(1))
        ranks = (sims > ts).sum(dim=1) + 1
        rf = ranks.float()
        return {
            "R@1": round(float((ranks <= 1).float().mean().item() * 100), 2),
            "R@5": round(float((ranks <= 5).float().mean().item() * 100), 2),
            "R@10": round(float((ranks <= 10).float().mean().item() * 100), 2),
            "MRR": round(float((1.0 / rf).mean().item() * 100), 2),
            "MedR": round(float(rf.median().item()), 1),
        }

    # 1. Real EEG
    real_embs, gts = [], []
    for b in all_b:
        ret = model.extract_brain_retrieval_embeddings(
            eeg_features=b["eeg_features"],
            eeg_attn_mask=b["eeg_attn_mask"],
            node_features=b["node_features"],
            word_coupled_edges=b["word_coupled_edges"],
            sent_connectome=b["sent_connectome"],
            rel_pos=b["rel_pos"],
            lex_pos=b["lex_pos"],
            word_salience=b["word_salience"],
            cog_load=b["cog_load"],
            task_id=b["task_id"],
        )
        real_embs.append(ret["post_rvq_emb"])
        gts.extend([sent_to_id[t] for t in b["target_text"]])
    R_emb = torch.cat(real_embs, dim=0)
    GT = torch.tensor(gts, dtype=torch.long)
    m_real = eval_retrieval(R_emb, GT)

    # 2. Gaze-Only Control: Zero out all EEG nodes & edges, keep ONLY 8-d oculomotor metadata
    gaze_embs = []
    for b in all_b:
        gaze_nodes = b["node_features"].clone()
        gaze_nodes[:, :, :480] = 0.0
        gaze_edges = torch.zeros_like(b["word_coupled_edges"])
        gaze_conn = torch.zeros_like(b["sent_connectome"])
        ret = model.extract_brain_retrieval_embeddings(
            eeg_features=torch.zeros_like(b["eeg_features"]),
            eeg_attn_mask=b["eeg_attn_mask"],
            node_features=gaze_nodes,
            word_coupled_edges=gaze_edges,
            sent_connectome=gaze_conn,
            rel_pos=b["rel_pos"],
            lex_pos=b["lex_pos"],
            word_salience=b["word_salience"],
            cog_load=b["cog_load"],
            task_id=b["task_id"],
        )
        gaze_embs.append(ret["post_rvq_emb"])
    G_emb = torch.cat(gaze_embs, dim=0)
    m_gaze = eval_retrieval(G_emb, GT)

    # 3. EEG-Shuffle Control: Permute EEG across trials while keeping true gaze sequence lengths
    torch.manual_seed(42)
    perm = torch.randperm(len(test_ds))
    shuf_samples = [test_ds.samples[i] for i in perm]
    shuf_loader = DataLoader(
        ZuCoEEGTextDataset(shuf_samples),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eeg_text,
    )
    shuf_embs = []
    for orig_b, shuf_b in zip(all_b, shuf_loader):
        m = orig_b["eeg_attn_mask"].unsqueeze(-1).float()
        ret = model.extract_brain_retrieval_embeddings(
            eeg_features=shuf_b["eeg_features"] * m,
            eeg_attn_mask=orig_b["eeg_attn_mask"],
            node_features=shuf_b["node_features"] * m,
            word_coupled_edges=shuf_b["word_coupled_edges"] * m,
            sent_connectome=shuf_b["sent_connectome"],
            rel_pos=orig_b["rel_pos"],
            lex_pos=orig_b["lex_pos"],
            word_salience=orig_b["word_salience"],
            cog_load=orig_b["cog_load"],
            task_id=orig_b["task_id"],
        )
        shuf_embs.append(ret["post_rvq_emb"])
    S_emb = torch.cat(shuf_embs, dim=0)
    m_shuf = eval_retrieval(S_emb, GT)

    # 4. Length-Matched Gaussian Noise Control (matched to true sequence lengths)
    torch.manual_seed(42)
    noise_embs = []
    for b in all_b:
        m = b["eeg_attn_mask"].unsqueeze(-1).float()
        ret = model.extract_brain_retrieval_embeddings(
            eeg_features=torch.randn_like(b["eeg_features"]) * m,
            eeg_attn_mask=b["eeg_attn_mask"],
            node_features=torch.randn_like(b["node_features"]) * m,
            word_coupled_edges=torch.randn_like(b["word_coupled_edges"]) * m,
            sent_connectome=torch.randn_like(b["sent_connectome"]),
            rel_pos=b["rel_pos"],
            lex_pos=b["lex_pos"],
            word_salience=b["word_salience"],
            cog_load=torch.randn_like(b["cog_load"]),
            task_id=b["task_id"],
        )
        noise_embs.append(ret["post_rvq_emb"])
    N_emb = torch.cat(noise_embs, dim=0)
    m_noise = eval_retrieval(N_emb, GT)

    results = {
        "theoretical_400way_chance": {"R@1": 0.25, "R@5": 1.25, "R@10": 2.50, "MRR": 1.64, "MedR": 200.5},
        "real_eeg": m_real,
        "gaze_only_control": m_gaze,
        "eeg_shuffle_control": m_shuf,
        "length_matched_noise_control": m_noise,
        "sanity_gap_vs_gaze_only_mrr": round(m_real["MRR"] - m_gaze["MRR"], 2),
        "sanity_gap_vs_eeg_shuffle_mrr": round(m_real["MRR"] - m_shuf["MRR"], 2),
        "sanity_gap_vs_noise_mrr": round(m_real["MRR"] - m_noise["MRR"], 2),
    }
    print(json.dumps(results, indent=2))
    out_f = "logs/q1_action3_neurorvq_bart_large_singletask/diagnostic_controls.json"
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    open(out_f, "w").write(json.dumps(results, indent=2))
