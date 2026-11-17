# Ratchaphon Khaemphukhiao
**Ph.D. Researcher in Biomedical Engineering · Brain-Computer Interfaces (BCI) & Neuro-Symbolic AI**

**Email:** `ratchaphon@rush0100.com` · **Phone:** (+66) 66 149 8641  
**Affiliation:** Department of Biomedical Engineering, School of Engineering, King Mongkut's Institute of Technology Ladkrabang (KMITL), Bangkok, Thailand  
**Academic Profiles:** [github.com/rush0100](https://github.com/rush0100) · [huggingface.co/RUSH0100](https://huggingface.co/RUSH0100) · [profile.rush0100.com](https://profile.rush0100.com)  
**Principal Advisor:** Assoc. Prof. Pattarapong Phasukkit, D.Eng. (`pattarapong.ph@kmitl.ac.th`, Dept. of Electronic Engineering, KMITL)  
**Target Visiting Institution:** Harbin Institute of Technology, Shenzhen (HIT Shenzhen) / Peng Cheng Laboratory

---

## Academic Executive Summary

Ph.D. candidate in Biomedical Engineering at King Mongkut's Institute of Technology Ladkrabang (KMITL) under the **KMITL Doctoral Scholarship (ทุนปริญญาเอกพระจอมเกล้าลาดกระบัง)**. Formally awarded the **KMITL Overseas Short-Term Research Mobility Grant (400,000 THB / ~82,000 RMB under Scholarship Clause 10.2)** to conduct visiting research abroad, with **Harbin Institute of Technology, Shenzhen (HIT Shenzhen)** designated as the primary desired research destination.

Ten years of end-to-end expertise spanning biosignal instrumentation (TI ADS1299 EEG analog front-end, high-density 105/128-ch EEG, mmWave radar), Riemannian manifold geometry, and large language model (LLM) architectures. Lead researcher of **RUSH-EEG2Text**, a fully implemented, teacher-forcing-free EEG-to-text decoding framework directly extending HIT Shenzhen's CET-MAE (ACL 2024) and BrainECHO (ACL 2025). Possesses dedicated local AI compute infrastructure (**NVIDIA DGX Spark**, Grace Blackwell GB10, 128 GB unified memory). Author of 5 peer-reviewed IEEE / BMEiCON papers; recipient of the **Platinum Award and Gold Medal with Jury Congratulations at BME IIF 2026 (Fukuoka, Japan)**.

*Personal Context & Research Passion:* Frequently visits Shenzhen (spouse's family domicile in Shenzhen). Deeply inspired by China's vibrant technological and scientific innovation ecosystem; visiting strictly for academic collaboration and joint research, not seeking commercial employment.

---

## Education

- **Ph.D. in Biomedical Engineering (In Progress, 2024 – Present)**  
  *King Mongkut's Institute of Technology Ladkrabang (KMITL), Bangkok, Thailand*  
  - **Scholarship:** KMITL Doctoral Scholarship (Full Tuition + Living Stipend + 400,000 THB Overseas Research Mobility Grant)  
  - **Advisor:** Assoc. Prof. Pattarapong Phasukkit, D.Eng.  
  - **Dissertation Focus:** Riemannian Connectome Modeling and Discrete Hierarchical Quantization for Non-Invasive Brain-to-Text Generative Interfaces.
- **M.Eng. in Electronic Engineering (2019 – 2023)**  
  *King Mongkut's Institute of Technology Ladkrabang (KMITL), Bangkok, Thailand*  
  - **Thesis:** Contactless physiological signal sensing and DSP-based feature extraction.
- **B.Eng. in Biomedical Engineering (2015 – 2019)**  
  *King Mongkut's Institute of Technology Ladkrabang (KMITL), Bangkok, Thailand*  
  - **Honors:** First Class Honors Candidate · Best Engineering Project Award (KMITL, 2019).

---

## Key Research Projects in BCI & Neuro-LLMs

### 1. RUSH-EEG2Text: Riemannian Connectome-Guided Hierarchical Quantization for Teacher-Forcing-Free Cross-Subject EEG-to-Text Decoding
*Lead Researcher & Developer (Under Review / Pre-Submission Draft for IEEE TNSRE / JBHI)*
- **Open-Science Resources:** [GitHub Code Repository](https://github.com/rush0100/RUSH-EEG2Text) · [Hugging Face Checkpoints](https://huggingface.co/RUSH0100/RUSH-EEG2Text-Qwen2.5-7B) · [Video Explainer](https://github.com/rush0100/RUSH-EEG2Text/raw/main/assets/rush-eeg2text-explained-simply.mp4)
- **Scientific Foundation:** Directly builds upon and extends the open research frontiers of **CET-MAE (ACL 2024)** and **BrainECHO (ACL 2025)** by Prof. Zhiguo Zhang and Dr. Zhenxi Song's group.
- **Key Breakthroughs on ZuCo 1.0 & 2.0 (18,276 trials, 30 subjects):**
  - **Fixation-Count Length Leakage Elimination:** Designed continuous [0, 1] scanpath-progress Fourier positional encoding, reducing pure Gaussian noise retrieval on unseen test subjects from 5.29% down to exact theoretical chance (MRR = 0.77%).
  - **3D 12-ROI Riemannian Connectome Fusion:** Parcellated 105 electrodes into 12 cortical ROIs across 4 reading phases (FFD, GD, TRT, TRT-FFD), fused with continuous 500-Hz Weighted Phase Lag Index (wPLI), Imaginary Coherence (iCoh), and Log-Euclidean Symmetric Positive Definite (SPD) tangent covariance via a zero-initialized Late Connectome Bottleneck Adapter.
  - **EMA-RVQ & Zero-Shot Retrieval:** 6-codebook Euclidean EMA-RVQ achieves **56.79 ± 2.44% zero-shot R@10** across 4 strictly unseen test subjects (ZMG, ZPH, YTL, YSL) in a 1,313-sentence pool, and **24.05% semantic cosine similarity** on 131 completely novel, disjoint sentences (double zero-shot split).
  - **Strict Teacher-Forcing-Free Decoding:** Beam-search (beams = 4) decoding with `Qwen2.5-7B-Instruct` (BF16 LoRA) reaches **BLEU-1 = 19.60%, BLEU-4 = 9.89%, ROUGE-1 F1 = 19.58%**, outperforming length-matched Gaussian noise by **+10.83% ΔBLEU-1 and +9.46% ΔBLEU-4 (23.0x signal-to-noise ratio)**.
  - **Controlled Baseline Parity:** Verified on `facebook/bart-large` (406M) under single-task `task1-SR` Non-TF decoding (BLEU-1 = 16.38%, BLEU-4 = 0.97%), outperforming recent baselines (LURS, EMNLP 2026).

### 2. NeuroReasoner: Privacy-Preserving Multimodal EEG-LLM for Cross-Center Clinical Assessment
*Lead Researcher & Developer (Active Research for Shenzhen Visiting Stay)*
- Multi-hospital federated parameter exchange (`FedAvg`) across 3 heterogeneous clinical cohorts:
  1. **OpenNeuro ds004504:** 88 subjects (AD, FTD, Cognitively Normal) with MMSE score prediction.
  2. **UNM & UIowa Parkinson's Cohorts:** Cross-center resting/task EEG (the exact benchmark from HIT Shenzhen's *EEG-MACS*, ACM MM 2024).
  3. **MODMA Clinical Depression:** 128-channel clinical EEG and 3-channel wearable EEG with PHQ-9 and GAD-7 regression.
- **Evidence-Grounding Bridge:** Predicts 11 quantitative EEG biomarkers (frontal alpha asymmetry, theta/beta ratio, Wernicke-Broca wPLI, Riemannian spectral power) into dedicated LLM tokens to enforce hallucination-free clinical chain-of-thought (`<think>...</think>`) reasoning under Local Differential Privacy (ε, δ).

### 3. 7Being: High-Density EEG & Physiological Biofeedback Platform
*Principal Architect & Investigator*
- Engineered end-to-end wearable and clinical biosignal acquisition architecture: TI ADS1299 analog front-end, Bluetooth Low Energy (BLE) streaming, real-time DSP band-power decomposition, and rPPG camera vital-sensing.
- Collected longitudinal EEG recordings during specialized meditation and cognitive tasks.
- **Honors:** Awarded **Platinum Award and Gold Medal with Jury Congratulations at BME IIF 2026 (Fukuoka, Japan)**.

---

## Professional & Engineering Leadership Experience (Industry-Academia Translation)

**Founder & CTO / CEO — SC AI Co., Ltd. & Kaset AI Co., Ltd.**  
*Pathum Thani, Thailand · Aug 2025 – Present*
- **7Being BCI Platform:** Engineered end-to-end BCI wellness and cognitive monitoring system: BLE firmware for EEG headsets and smart rings, on-device rPPG heart-rate computer vision, real-time EEG biofeedback, microservices backend, and international partner deployment across 6 countries (Bahrain, Jordan, Saudi Arabia, Oman, Indonesia, India).
- **Medical AI Solutions:** Developed DICOM viewer with intracranial hemorrhage detection and segmentation on brain CT scans; deployed multi-tenant enterprise RAG LLM platforms.
- **Grants & Accreditations:** Raised THB 1.8M in government competitive innovation grants (TED Fund THB 1.5M, SIBB THB 300K, both delivered and closed), THB 3.2M Google Cloud for Startups credits; member of NVIDIA Inception Program.

**Senior MLOps & Applied AI Infrastructure Consultant — Siametrics Consulting**  
*Bangkok, Thailand · Jan 2025 – Jul 2025*
- Lead technical architect for commercial bank's hyper-personalized AI marketing engine and the Stock Exchange of Thailand (SET ATLAS) automated financial intelligence system.
- Designed enterprise-grade model versioning, automated testing, and inference pipelines on NVIDIA A100/H100 clusters.

**Senior Machine Learning Infrastructure Engineer — VISAI.ai**  
*Bangkok, Thailand · May 2022 – Dec 2024*
- **National Telecom AI Infrastructure:** Designed and deployed the NVIDIA A100 multi-cluster microservices architecture powering National Telecom's AI Connect and VISAI AI Cloud Platform (`acp.visai.ai`).
- **Government AI Systems:** Engineered high-security NLP and automated inference infrastructure on NVIDIA A100 for Thailand's Central Investigation Bureau.
- Technical instructor on Docker and Kubernetes for state enterprise and government engineering teams.

**Research Assistant — CMKL University (Medical AI)**  
*Bangkok, Thailand · Aug 2019 – Aug 2021*
- Co-developed deep learning models for Retinopathy of Prematurity (ROP) screening with Queen Sirikit National Institute of Child Health, and AI heart-sound classification with KMITL Faculty of Medicine.

**Research Assistant — BURN LAB, KMITL**  
*Bangkok, Thailand · Aug 2015 – Aug 2019*
- Researched FPGA-based acoustic emission sensing with real-time AI classification for rail flaw detection; developed contactless fall detection using Kinect (national first prize in Chevron Young Makers Thailand).

---

## Peer-Reviewed Publications (IEEE / BMEiCON)

1. **R. Khaemphukhiao**, *et al.*, “Cardiorespiratory Agreement and Complementary Data Yield of a Piezoelectric Sensor and mmWave Radar,” in *Proc. International Conference on Biomedical Engineering (BMEiCON)*, 2026.
2. **R. Khaemphukhiao**, *et al.*, “Multimodal Assessment of a Center-Focused Meditation Protocol Using Frontal EEG Spectral Features and Wearable Physiology,” in *Proc. International Conference on Biomedical Engineering (BMEiCON)*, 2026.
3. **R. Khaemphukhiao**, *et al.*, “Segmentation of 24 GHz RF Respiration Signal Using GMM,” in *Proc. IEEE Biomedical Engineering International Conference (BMEiCON)*, 2019.
4. **R. Khaemphukhiao**, *et al.*, “Using Doppler Radar to Classify Respiration by MFCC and Deep Neural Networks,” in *Proc. IEEE International Conference on Electrical Engineering/Electronics, Computer, Telecommunications and Information Technology (ECTI-CON)*, 2019.
5. **R. Khaemphukhiao**, *et al.*, “Comparison of LoRa 915 MHz and 433 MHz Propagation on Distance Coverage,” in *Proc. IEEE International Conference on Inventive Systems and Control (ICISC)*, 2018.

*Manuscripts in Preparation / Pre-Submission:*
- **R. Khaemphukhiao**, [Invited Co-Author Slots: Z. Song, Z. Zhang], P. Phasukkit, “*RUSH-EEG2Text: Riemannian Connectome-Guided Hierarchical Quantization for Teacher-Forcing-Free Cross-Subject EEG-to-Text Decoding*,” Target: *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*, 2026.
- **R. Khaemphukhiao**, P. Phasukkit, “*NeuroReasoner: Privacy-Preserving Multimodal EEG-LLM with Chain-of-Thought Neural Evidence Grounding for Cross-Center Cognitive and Affective Assessment*,” Target: *IEEE Journal of Biomedical and Health Informatics (JBHI)*, 2026.

---

## Research Grants & Academic Honors

- **KMITL Doctoral Scholarship Overseas Research Mobility Grant** — Awarded **400,000 THB (~82,000 RMB)** dedicated funding to conduct 3–6 months visiting research abroad (Clause 10.2).
- **Platinum Award & Gold Medal with Jury Congratulations** — BME IIF 2026, Fukuoka, Japan (Brain-Computer Interface sensing & biofeedback).
- **First Prize National Winner, Young Makers Thailand** — Chevron Enjoy Science / National Science Museum (400,000 THB research prize).
- **Best Engineering Project Award** — Faculty of Engineering, KMITL, 2019.
- **Google Professional Machine Learning Engineer Certification** — Certified ML Specialist.
- **Certified Kubernetes Administrator (CKA)** — Cloud Native Computing Foundation (CNCF).
- **Member, NVIDIA Inception Program** — Local access to enterprise AI compute acceleration.

---

## Technical & Experimental Proficiencies

| Domain | Technical Capabilities |
| :--- | :--- |
| **BCI & Neuroimaging** | High-density scalp EEG (105-ch ZuCo, 128-ch HydroCel, 64-ch 10-20), EGI NetStation, MNE-Python, EEGLAB, 3D anatomical ROI parcellation, artifact channel masking, unsupervised robust normalization. |
| **Connectomics & Riemannian Geometry** | Symmetric Positive Definite ($\mathcal{S}_{++}^n$) manifold mapping, Log-Euclidean tangent space projections, Welch Cross-Spectral Density (CSD), Weighted Phase Lag Index (wPLI), Imaginary Coherence (iCoh), phase-amplitude coupling. |
| **Deep Learning & Neuro-LLMs** | PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), Causal decoders (`Qwen2.5-7B`, `Llama-3`), Seq2Seq (`BART-large`), Residual Vector Quantization (EMA-RVQ), Perceiver resamplers, InfoNCE multi-positive contrastive learning. |
| **Biosignal Instrumentation** | Analog Front-End design (TI ADS1299 EEG, ADS1292), BLE real-time telemetry, 24 GHz & 60 GHz mmWave FMCW radar, DSP filtering, FPGA/Zynq embedded processing. |
| **Compute & Infrastructure** | On-premise **NVIDIA DGX Spark** (Grace Blackwell GB10, 128 GB UMA), NVIDIA A100/H100 clusters, Linux/Docker, distributed model training, mixed-precision (BF16/FP16). |
| **Languages** | **English:** Professional academic & technical proficiency (fluent in research writing and oral presentation) · **Thai:** Native · **Chinese:** Basic everyday conversational. |
