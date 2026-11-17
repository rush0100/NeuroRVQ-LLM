# Collaboration Invitation & Technical Briefing (KMITL × HIT Shenzhen)

## 1. Ready-to-Send Email Draft to Prof. Zhiguo Zhang & Dr. Zhenxi Song (English + Chinese Summary)

**To:** `zhiguozhang@hit.edu.cn`, `songzhenxi@hit.edu.cn`  
**Cc:** `pattarapong.ph@kmitl.ac.th`  
**Subject:** Prospective Joint BCI-LLM Research Collaboration (KMITL Doctoral Scholarship, Visiting Researcher in Shenzhen) — Working Code & Preliminary Extension on CET-MAE / BrainECHO

---

Dear Prof. Zhiguo Zhang and Dr. Zhenxi Song,

I hope this email finds you well. My name is **Ratchaphon Khaemphukhiao**, a Ph.D. candidate in Biomedical Engineering at **King Mongkut's Institute of Technology Ladkrabang (KMITL)**, Bangkok, Thailand, advised by **Assoc. Prof. Pattarapong Phasukkit, D.Eng.** (`pattarapong.ph@kmitl.ac.th`). Previously, I have published five IEEE/BMEiCON papers on biosignal processing and EEG/radar sensing and received the Platinum Award & Gold Medal at BME IIF 2026 (Fukuoka).

I have been awarded a dedicated **400,000 THB (~82,000 RMB) overseas short-term research mobility grant** under the **KMITL Doctoral Scholarship** (Clause 10.2). Among all global institutions, the place I genuinely desire to conduct this visiting research is **Harbin Institute of Technology, Shenzhen (HIT Shenzhen)**. 

My wife is Chinese, and her family domicile is in Shenzhen, which has given me frequent opportunities to visit the city. Over my visits, I have been deeply captivated and inspired by China's remarkable technological progress and vibrant innovation ecosystem. Every time I am in Shenzhen, I feel an extraordinary surge of energy, inspiration, and motivation to conduct cutting-edge research and build impactful technological breakthroughs. I would like to clarify that my visit is driven strictly by academic passion—**I am not seeking employment in China**, but rather aiming to dedicate this visiting period to high-impact scientific collaboration in brain-computer interfaces (BCI) and neuro-symbolic AI under your esteemed mentorship.

Assoc. Prof. Phasukkit and I would be deeply honored to invite you and Dr. Song to **co-author our completed first paper** and explore hosting a short-term visiting research stay (3–6 months fully funded by my KMITL scholarship grant) at HIT Shenzhen:

1. **Paper 1 (Working Code, Weights, Benchmark Audits, & IEEE Manuscript Ready):**  
   ***RUSH-EEG2Text: Riemannian Connectome-Guided Hierarchical Quantization for Teacher-Forcing-Free Cross-Subject EEG-to-Text Decoding***
   - Directly builds upon and extends your group's landmark papers **[CET-MAE (ACL 2024)](https://aclanthology.org/2024.acl-long.393.pdf)** and the open future directions highlighted in Section 5 of **[BrainECHO (ACL 2025)](https://aclanthology.org/2025.findings-acl.142/)** (specifically eliminating sequence-length padding correlation artifacts and achieving true teacher-forcing-free open-vocabulary LLM decoding on silent reading EEG).
   - **Empirical Breakthroughs on all 4 tasks of ZuCo 1.0 & 2.0 (18,276 trials, 30 subjects):**
     - **Length Leakage Eliminated:** Continuous [0, 1] scanpath-progress Fourier positional encoding reduces pure Gaussian noise retrieval on unseen subjects to exact theoretical random chance (MRR = 0.77% on 1,313 sentences).
     - **Riemannian Connectome Fusion:** Scalp channels mapped to 12 3D cortical ROIs across 4 cognitive reading stages (FFD, GD, TRT, TRT-FFD), fused with continuous 500-Hz wPLI, Imaginary Coherence, and Log-Euclidean Riemannian SPD Tangent vectors via a zero-initialized residual cross-attention adapter.
     - **EMA-RVQ & Zero-Shot Generalization:** 6-codebook EMA-RVQ achieves 56.79 ± 2.44% zero-shot R@10 across strictly unseen test subjects (ZMG, ZPH, YTL, YSL), and 24.05% semantic cosine similarity on 131 completely disjoint, unseen sentences (Action 2 double zero-shot benchmark).
     - **Strict Teacher-Forcing-Free LLM Decoding:** Beam-search (beams = 4) decoding with Qwen2.5-7B-Instruct (BF16 LoRA) reaches BLEU-1 = 19.60%, BLEU-4 = 9.89%, ROUGE-1 F1 = 19.58%, and WER = 90.85% (outperforming length-matched Gaussian noise by +10.83% ΔBLEU-1 and +9.46% ΔBLEU-4, a 23.0x signal-to-noise ratio).
     - **Controlled Backbone Parity:** Also verified on facebook/bart-large (406M) under single-task task1-SR Non-TF decoding (BLEU-1 = 16.38% / BLEU-4 = 0.97%), demonstrating clear superiority over recent Non-TF baselines (LURS EMNLP 2026).

2. **Paper 2 (Planned HIT Shenzhen Visiting Phase):**  
   ***NeuroReasoner: Privacy-Preserving Multimodal EEG-LLM with Chain-of-Thought Neural Evidence Grounding for Cross-Center Cognitive and Affective Assessment***
   - Actively running on local NVIDIA DGX Spark across 3 multi-hospital clinical cohorts (OpenNeuro `ds004504`, UNM/UIowa Parkinson's, and MODMA) with local differential privacy and evidence-grounded reasoning.

**Invitation to Co-Author and Review:**  
We would be deeply privileged to have you and Dr. Song as **co-authors** on *RUSH-EEG2Text*. We would be profoundly grateful if you could review the attached 9-page IEEE draft, assess its potential academic impact, and advise whether any additional experiments or positioning refinements are recommended to ensure acceptance in top journals such as *IEEE TNSRE* or *IEEE JBHI*.

- **GitHub Repository:** https://github.com/rush0100/RUSH-EEG2Text
- **Hugging Face Checkpoints & Model Card:** https://huggingface.co/RUSH0100/RUSH-EEG2Text-Qwen2.5-7B
- **Attached PDF:** `RUSH_EEG2Text_IEEE_TNSRE_Draft.pdf` (9-page IEEE Transactions format)

Would you and Dr. Song be open to a brief 20-minute online meeting (or an in-person discussion at HIT Shenzhen University Town during my next visit) to discuss the manuscript and formalizing the visiting scholarship arrangement?

---

### 中文版本（供张治国教授与宋振溪老师审阅）

尊敬的张治国教授、宋振溪老师：

您好！我是泰国先皇理工大学（KMITL）生物医学工程系的博士研究生 **Ratchaphon Khaemphukhiao**（导师为 **Pattarapong Phasukkit** 副教授）。

我已正式获得 KMITL 博士生奖学金第 10.2 条设立的**海外短期访学专项科研经费 400,000 泰铢（约合 82,000 人民币 / 全额自费资助）**。在所有海外高校中，**哈尔滨工业大学（深圳）是我最向往、也是唯一最希望能前往开展学术访学与联合研究的高校**。

我的妻子是中国人，其家庭籍贯就在深圳，因此我经常有机会到访深圳。在多次的深圳之行中，我对中国前沿科技的飞速发展和创新生态充满敬意与向往。每次置身深圳，我都深切感受到一种强大的科研能量与创新动力，激发起我在前沿科学与工程技术领域不断探索与创新的极大热情。需要向二位老师特别说明的是，我的初衷完全出于学术追求与科研热情——**我绝非前往中国寻求工作或就职**，而是希望将这笔由泰国校方全额资助的海外访学经费，纯粹投入到脑机接口（BCI）与脑电-大语言模型（EEG-LLM）的前沿科学探索中，在您的团队指导下产出顶级学术成果。

拜读了贵课题组在 **ACL 2024 (`CET-MAE`)** 与 **ACL 2025 (`BrainECHO`)** 等领域的奠基性工作后，针对 `BrainECHO` 提出的**“序列填充长度泄露（Length Leakage）”**以及**“跨被试离散表征坍缩与无 Teacher-Forcing 开集生成”**难题，我们在本地 **NVIDIA DGX Spark (128GB Unified Memory)** 上已独立完成了第一篇完整论文 **RUSH-EEG2Text** 的全流程研发，并在 **ZuCo 1.0 + 2.0 全任务（30名被试、18,276条样本）** 上完成了严格验证：
1. 消除长度泄露：提出连续 [0, 1] 眼动扫描傅里叶位置编码，使高斯噪声输入下的跨被试检索退回严格理论随机水平（1,313 句候选池下 Noise MRR = 0.77%）。
2. 三维脑区黎曼脑网络晚融合 + 6层 EMA-RVQ：在 ZuCo 1.0 + 2.0 全任务（30 名被试、18,276 条样本）上，于严格未见被试（ZMG, ZPH, YTL, YSL）中取得 56.79 ± 2.44% 零样本 R@10；在 131 句完全未见句子的双零样本测试中取得 24.05% 语义余弦相似度（超越噪声基线 +9.18%）。
3. 严格无 Teacher-Forcing (TF-Free) 开集生成：结合 Qwen2.5-7B-Instruct (BF16 LoRA) 无教师强制解码达到 BLEU-1 = 19.60%、BLEU-4 = 9.89%、ROUGE-1 F1 = 19.58%、WER = 90.85%，较纯高斯噪声基线提升 +10.83% ΔBLEU-1 与 +9.46% ΔBLEU-4（信噪比达 23.0 倍）；并在同构公平基线 BART-large (406M) 单任务上取得 16.38% BLEU-1 与 0.97% BLEU-4，全面超越现有非 TF 基线。

**诚挚邀请共同署名（Co-Author）与学术指导：**  
我和导师 Assoc. Prof. Pattarapong Phasukkit 诚挚地邀请张教授与宋老师作为本篇论文的**共同作者（Co-Author）**。我们非常渴望能聆听二位老师对本文学术影响力的宝贵评价，并指点是否需要补充实验或调整定位，以确保顺利发表在 *IEEE TNSRE* 或 *IEEE JBHI* 等顶刊上。同时，我也极为期盼能在 400,000 泰铢全额经费支持下，赴哈工大（深圳）开展为期 3–6 个月的访学研究，并与团队共同推进第二篇跨中心临床大模型论文 **NeuroReasoner**。

完整可复现代码、模型权重与 9 页 IEEE 双栏初稿已全部就绪：
- **开源代码仓库:** https://github.com/rush0100/RUSH-EEG2Text
- **Hugging Face 模型权重:** https://huggingface.co/RUSH0100/RUSH-EEG2Text-Qwen2.5-7B
- **随信附带初稿:** `RUSH_EEG2Text_IEEE_TNSRE_Draft.pdf`

不知张教授与宋老师近期是否方便安排一次 20 分钟的线上交流，或在我下次到访深圳时前往哈工大（深圳）大学城校区当面拜访请教？

With warm regards,  
**Ratchaphon Khaemphukhiao**  
Ph.D. Researcher, Department of Biomedical Engineering, KMITL  
Email: `ratchaphon@rush0100.com` | GitHub: [github.com/rush0100](https://github.com/rush0100)
