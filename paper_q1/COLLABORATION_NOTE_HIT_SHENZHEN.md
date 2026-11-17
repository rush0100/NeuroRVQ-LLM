# Collaboration Invitation & Technical Briefing (KMITL × HIT Shenzhen)

## 1. Ready-to-Send Email Draft to Prof. Zhiguo Zhang & Dr. Zhenxi Song (English + Chinese Summary)

**To:** `zhiguozhang@hit.edu.cn`, `songzhenxi@hit.edu.cn`  
**Cc:** `pattarapong.ph@kmitl.ac.th`  
**Subject:** Prospective Joint BCI-LLM Research Collaboration (KMITL Doctoral Scholarship, Visiting Researcher in Shenzhen) — Working Code & Preliminary Extension on CET-MAE / BrainECHO

---

Dear Prof. Zhiguo Zhang and Dr. Zhenxi Song,

I hope this email finds you well. My name is **Ratchaphon Khaemphukhiao**, a Ph.D. researcher in Biomedical Engineering at **King Mongkut's Institute of Technology Ladkrabang (KMITL)**, Bangkok, Thailand, advised by **Assoc. Prof. Pattarapong Phasukkit, D.Eng.** (`pattarapong.ph@kmitl.ac.th`). Previously, I have published five IEEE/BMEiCON papers on biosignal processing and EEG/radar sensing and received the Platinum Award & Gold Medal at BME IIF 2026 (Fukuoka). My wife is from Shenzhen, and I spend regular time in Shenzhen with dedicated local AI compute infrastructure (**NVIDIA DGX Spark**, 128 GB Grace Blackwell unified memory).

Under the **KMITL Doctoral Scholarship Program** (which includes dedicated KMITL funding for short-term overseas collaborative research in Shenzhen), Assoc. Prof. Phasukkit and I would be deeply honored to explore a joint research collaboration with your group at **Harbin Institute of Technology, Shenzhen**, targeting two high-impact Q1 journal papers:

1. **Paper 1 (Working Code, 14-Run Ablation, & IEEE Manuscript Draft Ready):**  
   ***NeuroRVQ-LLM: Connectome-Guided Residual Vector Quantization for Length-Invariant and Teacher-Forcing-Free Cross-Subject EEG-to-Text Decoding***
   - Directly inspired by your group's **[CET-MAE (ACL 2024)](https://aclanthology.org/2024.acl-long.393.pdf)** and the open future directions noted in Section 5 of **[BrainECHO (ACL 2025)](https://aclanthology.org/2025.findings-acl.142/)** (specifically eliminating signal-length padding correlations and achieving teacher-forcing-free open-vocabulary LLM decoding on silent reading EEG).
   - **What we have built and verified so far on all 4 tasks of ZuCo 1.0 & 2.0 (18,276 trials, 30 subjects):**
     - Identified and eliminated fixation-count length leakage via continuous `[0,1]` scanpath-progress Fourier positional encoding (reducing pure Gaussian noise retrieval on unseen subjects from `5.29%` down to exact random chance `1.42%` / `0.77%`).
     - Mapped the 105 scalp channels (`chanlocs105.mat`) into 12 3D cortical ROIs across four cognitive reading windows (`FFD, GD, TRT, TRT-FFD`), paired with a **Zero-Initialized Late Connectome Bottleneck Adapter** fusing 500-Hz Welch **wPLI**, **Imaginary Coherence**, and **Log-Euclidean Riemannian SPD Tangent** vectors.
     - Designed an un-inflated 6-codebook **EMA-RVQ** with multi-positive cross-subject InfoNCE, achieving **`56.09%` zero-shot R@10 (median rank `9.0` out of 1,313 unique sentences, `130×` random chance R@1)** across 4 strictly unseen test subjects (`ZMG`, `ZPH`, `YTL`, `YSL`).
     - In Stage 2 strict **Teacher-Forcing-Free** open-vocabulary generation with `Qwen2.5-7B-Instruct` (BF16 LoRA + 50% scheduled target-word dropout), achieved **`19.60%` BLEU-1, `9.89%` BLEU-4, `19.58%` ROUGE-1 F1, and `90.85%` WER** on the 2,717 unseen-subject test trials, outperforming pure Gaussian noise (`8.77%` BLEU-1, `0.43%` BLEU-4, `182.92%` WER) by **`+10.83%` $\Delta$BLEU-1 and `+9.46%` $\Delta$BLEU-4 (`23.0×` over noise)** and surpassing recent cross-subject Non-TF baselines (`LURS` EMNLP 2026: `15.39%` BLEU-1 / `0.89%` BLEU-4).

2. **Paper 2 (Planned Shenzhen Visiting Phase — Extending *EEG-MACS* & *GCAN*):**  
   ***NeuroReasoner: Privacy-Preserving Multimodal EEG-LLM with Chain-of-Thought Neural Evidence Grounding for Cross-Center Cognitive and Affective Assessment***
   - Reuses the exact privacy-preserving 12-ROI wPLI / Riemannian SPD biomarkers from Paper 1 to ground hallucination-free `<think>...</think>` clinical reasoning on a local DGX Spark appliance across heterogeneous hospital EEG montages.

To make reviewing effortless, we have organized our full reproducible codebase, 14-run ablation logs, and a 7-page IEEE Transactions draft here:
- **GitHub Repository:** `https://github.com/rush0100/NeuroRVQ-LLM`
- **Hugging Face Checkpoints & Model Card:** `https://huggingface.co/RUSH0100/NeuroRVQ-LLM-Qwen2.5-7B`
- **Compiled IEEE Draft PDF:** `paper_q1/NeuroRVQ_LLM_IEEE_TNSRE_Draft.pdf` (also attached)

Would you and Dr. Song be open to a brief 20-minute online meeting (or an in-person visit at HIT Shenzhen University Town at your convenience) to discuss aligning this manuscript and a short-term visiting research plan with your lab's roadmap?

---

### 中文摘要（供张治国教授与宋振溪老师审阅）

尊敬的张治国教授、宋振溪老师：

您好！我是泰国先皇理工大学（KMITL）生物医学工程系的博士研究生 **Ratchaphon Khaemphukhiao**（导师为 **Pattarapong Phasukkit** 副教授）。我的妻子是深圳人，同时我受 KMITL 博士奖学金（含海外短期访学专项经费）资助，非常希望能有机会与哈工大（深圳）团队开展联合科研合作。

拜读了贵课题组在 **ACL 2024 (`CET-MAE`)**、**ACL 2025 (`BrainECHO`)** 及 **ACM MM 2024 (`EEG-MACS`)** 的系列前沿工作后，我们针对 `BrainECHO` 第五节指出的**“序列填充长度泄露（Length Leakage）”**以及**“静默阅读缺乏声学中间模态时的跨被试离散化坍缩与 Teacher-Forcing 依赖”**问题，在本地 **NVIDIA DGX Spark (128GB 统一内存)** 上完成了第一篇合作论文原型 **NeuroRVQ-LLM** 的代码实现与全量 **ZuCo 1.0 + 2.0（4个任务、30名被试、18,276条样本）** 实验验证：
1. **消除长度泄露**：提出连续 `[0,1]` 眼动扫描进度傅里叶位置编码，使纯高斯噪声输入下的跨被试检索退回理论随机水平（1,313句候选池下 Noise MRR = `0.77%`）。
2. **三维脑区多阶段功率 + 晚融合脑网络瓶颈（wPLI / iCoh / 黎曼流形 SPD 切空间）+ 6层无膨胀 EMA-RVQ**：在严格未见被试（Unseen Subjects）的 **1,313句零样本检索**中达到 **`R@10 = 56.09%`（中位排名第 `9.0` 名，`R@1` 达随机水平的 `130` 倍）**；在第二阶段 `Qwen2.5-7B` 严格**无 Teacher-Forcing（TF-Free）**生成评估（2,717条未见被试测试样本）中达到 **`BLEU-1 = 19.60%`、`BLEU-4 = 9.89%`、`ROUGE-1 F1 = 19.58%`、`WER = 90.85%`**，显著超越纯高斯噪声对照（`BLEU-4 = 0.43%`，净增益 **`+10.83%` $\Delta$BLEU-1 / `+9.46%` $\Delta$BLEU-4**）及最新跨被试非 Teacher-Forcing 基线。

完整开源代码、14组消融实验日志及 IEEE Transactions 双栏初稿已整理在 GitHub 仓库中。诚挚期待能与您和宋老师线上简短交流，或在我到访深圳期间前往哈工大（深圳）当面请教！

With warm regards,  
**Ratchaphon Khaemphukhiao**  
Ph.D. Researcher, Department of Biomedical Engineering, KMITL  
Email: `ratchaphon@rush0100.com` | GitHub: [github.com/rush0100](https://github.com/rush0100)
