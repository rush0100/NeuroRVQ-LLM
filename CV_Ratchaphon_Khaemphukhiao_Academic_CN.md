# 个人学术简历 (Academic CV)

**Ratchaphon Khaemphukhiao (รัชพล เข็มภูเขียว)**  
**博士研究生 · 脑机接口 (BCI) 与神经大语言模型 (Neuro-Symbolic AI)**

- **电子邮箱:** `ratchaphon@rush0100.com`
- **联系电话:** (+66) 66 149 8641
- **所在院校:** 泰国先皇理工大学（KMITL）工程学院生物医学工程系，曼谷，泰国
- **主要学术主页:** [github.com/rush0100](https://github.com/rush0100) · [huggingface.co/RUSH0100](https://huggingface.co/RUSH0100) · [profile.rush0100.com](https://profile.rush0100.com)
- **博士生导师:** Assoc. Prof. Pattarapong Phasukkit, D.Eng. (`pattarapong.ph@kmitl.ac.th`, KMITL 电子工程系)
- **意向访学高校与团队:** 哈尔滨工业大学（深圳）计算机科学与技术学院 / 鹏城实验室（张治国教授、宋振溪博士团队）

---

## 个人简介与科研访学陈述 (Executive Summary)

本人为泰国先皇理工大学（KMITL）生物医学工程系博士研究生，师从 Pattarapong Phasukkit 副教授，享受 **KMITL 博士全额奖学金**。目前已正式获批 **KMITL 博士生海外短期访学专项科研基金（400,000 泰铢 / 约合 82,000 元人民币，全额资助）**，专门用于支持赴海外高水平科研机构开展为期 3–6 个月的联合科研访学。在所有全球高校中，**哈尔滨工业大学（深圳）为本人唯一的意向访学合作单位**。

本人深耕生物医学信号传感（TI ADS1299 脑电模拟前端芯片、105/128 导高密度脑电、毫米波雷达）、黎曼流形几何算法与大语言模型（LLM）解码技术十余年。针对贵团队在 **CET-MAE (ACL 2024)** 与 **BrainECHO (ACL 2025)** 中提出的开放前沿方向，本人已独立主导完成了第一篇合作论文原型 **RUSH-EEG2Text** 的全流程研发、消融实验与 9 页 IEEE Transactions 论文草稿，在本地 **NVIDIA DGX Spark 工作站 (Grace Blackwell GB10, 128GB 统一内存)** 上完整复现并取得了显著优于同类基线的零样本与无 Teacher-Forcing 解码性能。迄今已发表 5 篇 IEEE / BMEiCON 国际同行评议会议论文，并荣获 **BME IIF 2026（日本福冈）国际发明展白金特等奖与评审团祝贺金奖**。

*个人背景与访学动力：*  
我的妻子是中国人，其家庭籍贯就在深圳，因此我经常往返于曼谷与深圳。在多次到访深圳的过程中，我对中国前沿科技的飞速发展和创新生态充满敬意与向往，每次置身深圳都能深切感受到强烈的科研能量与探索动力。在此特别向二位老师澄清：**本人访学初衷完全出于纯粹的学术热爱与国际合作，绝非赴华寻求商业就职或工作**，将携泰国校方全额资助的 400,000 泰铢专项科研经费前往哈工大（深圳），全力以赴产出高水平学术成果。

---

## 教育背景

- **工学博士 (Ph.D. in Biomedical Engineering) — 在读 (2024 – 至今)**  
  *泰国先皇理工大学 (KMITL)，工程学院生物医学工程系*  
  - **奖学金:** KMITL 博士全额奖学金（免全额学费 + 生活津贴 + 400,000 泰铢海外访学专项资助）  
  - **导师:** Assoc. Prof. Pattarapong Phasukkit, D.Eng.  
  - **研究方向:** 黎曼脑功能网络连接组建模与分层残差矢量量化在无创脑电文本生成大模型中的应用
- **工学硕士 (M.Eng. in Electronic Engineering) (2019 – 2023)**  
  *泰国先皇理工大学 (KMITL)，工程学院电子工程系*  
  - **硕士论文:** 基于非接触生理传感与数字信号处理的生物特征提取技术
- **工学学士 (B.Eng. in Biomedical Engineering) (2015 – 2019)**  
  *泰国先皇理工大学 (KMITL)，工程学院生物医学工程系*  
  - **荣誉:** 一等荣誉学士候选人 · KMITL 2019 年度最佳工程科研设计奖 (Best Engineering Project Award)

---

## 代表性脑机接口与神经大模型科研成果

### 1. RUSH-EEG2Text: 基于黎曼连接组引导的分层量化无 Teacher-Forcing 跨被试脑电转文本系统
*主要研究者与系统架构师（IEEE TNSRE / JBHI 待投稿完整初稿已就绪）*
- **开源成果库:** [GitHub 源码](https://github.com/rush0100/RUSH-EEG2Text) · [Hugging Face 模型权重](https://huggingface.co/RUSH0100/RUSH-EEG2Text-Qwen2.5-7B) · [演示视频](https://github.com/rush0100/RUSH-EEG2Text/raw/main/assets/rush-eeg2text-explained-simply.mp4)
- **学术承接关系:** 紧扣哈工大深圳团队在 **CET-MAE (ACL 2024)** 与 **BrainECHO (ACL 2025)** 中提出的“静默阅读缺乏声学中间目标”以及“序列填充长度泄露（Length Leakage）”等关键开放难题。
- **核心技术突破（验证于 ZuCo 1.0 + 2.0 全部 4 个任务，30 名被试、18,276 条样本）:**
  1. **彻底消除长度泄露:** 提出连续 [0, 1] 眼动扫描傅里叶位置编码，使高斯噪声输入下的跨被试检索退回严格理论随机水平（1,313 句候选池下 Noise MRR = 0.77%）。
  2. **三维脑区黎曼脑网络晚融合 + 6 层 EMA-RVQ:** 将 105 导电极映射至 12 个解剖皮层 ROI，提取 4 个认知阅读阶段（FFD, GD, TRT, TRT-FFD）功率，与 500-Hz 连续波形计算出的加权相位滞后指数（wPLI）、虚部相干性（iCoh）以及对数欧氏对称正定矩阵（Log-Euclidean SPD）黎曼切空间协方差特征晚融合。在严格未见被试（ZMG, ZPH, YTL, YSL）的 1,313 句零样本检索中达到 **56.79 ± 2.44% R@10**；在 131 句完全未见句子的双零样本测试中取得 **24.05% 语义余弦相似度（超越噪声基线 +9.18%）**。
  3. **严格无 Teacher-Forcing 开集生成:** 结合 `Qwen2.5-7B-Instruct`（BF16 LoRA）无教师强制解码达到 **BLEU-1 = 19.60%、BLEU-4 = 9.89%、ROUGE-1 F1 = 19.58%**，较纯高斯噪声基线提升 **+10.83% ΔBLEU-1 与 +9.46% ΔBLEU-4（信噪比达 23.0 倍）**；并在同构公平基线 `BART-large` (406M) 单任务上取得 **16.38% BLEU-1**，全面超越现有非 TF 基线。

### 2. NeuroReasoner: 隐私保护与神经证据锚定的多中心跨模态脑电大模型
*主要研究者与系统架构师（为深圳访学阶段推进的核心课题）*
- 在本地 NVIDIA DGX Spark 上构建支持跨医院联邦参数聚合（`FedAvg` + 本地差分隐私）的多中心临床大模型架构：
  1. **OpenNeuro ds004504:** 88 名阿尔茨海默病（AD）、额颞叶痴呆（FTD）及健康对照被试与 MMSE 评分预测。
  2. **UNM 与 UIowa 帕金森氏症数据集:** 跨中心静息态/任务态脑电（贵团队发表于 *ACM MM 2024 (EEG-MACS)* 的原班基准）。
  3. **MODMA 临床抑郁症数据集:** 包含 128 导临床脑电与 3 导便携式脑电，支持 PHQ-9 与 GAD-7 量表回归。
- **证据锚定桥（Evidence Grounding Bridge）:** 将额叶不对称性、Theta/Beta 功率比、Wernicke-Broca wPLI 相位连接等 11 项定量脑电生物标志物显式投射为专用提示词 Token，确保大模型临床思维链（`<think>...</think>`）推理具备真实神经生理证据支撑，杜绝虚假幻觉。

### 3. 7Being: 高密度脑电与多维生理反馈平台
*核心技术架构师*
- 自主设计并研发从 TI ADS1299 脑电模拟前端、低功耗蓝牙（BLE）固件、多频带功率谱实时分解到底层微服务的全栈软硬件系统。
- 收集专业冥想及认知任务下的长时程多模态脑电信号。
- **获奖荣誉:** 荣获 **2026 年日本福冈 BME IIF 国际发明展白金特等奖与评审团祝贺金奖**。

---

## 国际学术论文 (IEEE / BMEiCON)

1. **R. Khaemphukhiao**, et al., “Cardiorespiratory Agreement and Complementary Data Yield of a Piezoelectric Sensor and mmWave Radar,” in *Proc. International Conference on Biomedical Engineering (BMEiCON)*, 2026.
2. **R. Khaemphukhiao**, et al., “Multimodal Assessment of a Center-Focused Meditation Protocol Using Frontal EEG Spectral Features and Wearable Physiology,” in *Proc. International Conference on Biomedical Engineering (BMEiCON)*, 2026.
3. **R. Khaemphukhiao**, et al., “Segmentation of 24 GHz RF Respiration Signal Using GMM,” in *Proc. IEEE Biomedical Engineering International Conference (BMEiCON)*, 2019.
4. **R. Khaemphukhiao**, et al., “Using Doppler Radar to Classify Respiration by MFCC and Deep Neural Networks,” in *Proc. IEEE International Conference on Electrical Engineering/Electronics, Computer, Telecommunications and Information Technology (ECTI-CON)*, 2019.
5. **R. Khaemphukhiao**, et al., “Comparison of LoRa 915 MHz and 433 MHz Propagation on Distance Coverage,” in *Proc. IEEE International Conference on Inventive Systems and Control (ICISC)*, 2018.

*在投 / 待投稿论文 (Pre-Submission):*
- **R. Khaemphukhiao**, [特邀共同作者席位: 宋振溪博士, 张治国教授], P. Phasukkit, “*RUSH-EEG2Text: Riemannian Connectome-Guided Hierarchical Quantization for Teacher-Forcing-Free Cross-Subject EEG-to-Text Decoding*,” 拟投期刊: *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*, 2026.
- **R. Khaemphukhiao**, P. Phasukkit, “*NeuroReasoner: Privacy-Preserving Multimodal EEG-LLM with Chain-of-Thought Neural Evidence Grounding for Cross-Center Cognitive and Affective Assessment*,” 拟投期刊: *IEEE Journal of Biomedical and Health Informatics (JBHI)*, 2026.

---

## 科研资助与个人荣誉

- **KMITL 博士生海外短期访学专项科研资助** — 获批 **400,000 泰铢（约合 82,000 人民币）** 全额专项资助（条款 10.2）。
- **国际发明展白金特等奖与评审团祝贺金奖 (Platinum Award & Gold Medal)** — 2026 年日本福冈 BME IIF（脑机接口传感与神经反馈）。
- **泰国青年发明家大赛全国第一名 (National Winner)** — Chevron Enjoy Science / 泰国国家科技馆 (奖金 400,000 泰铢)。
- **KMITL 最佳工程科研设计大奖** — KMITL 工程学院, 2019。
- **Google 认证专业机器学习工程师 (Google Professional ML Engineer)**。
- **Linux 基金会认证 Kubernetes 管理员 (CKA)**。
- **NVIDIA Inception Program 会员** — 本地科研具备企业级 AI 算力生态支持。

---

## 个人专业技能与实验能力

- **脑电与脑成像分析:** 高密度头皮脑电（105 导 ZuCo、128 导 HydroCel、64 导 10-20）、EGI NetStation 导联系统、MNE-Python、EEGLAB、三维解剖皮层 ROI 映射、坏道掩膜、无监督被试自适应校准。
- **脑功能连接与黎曼几何:** 对称正定矩阵（SPD）黎曼流形投影、对数欧氏（Log-Euclidean）切空间映射、Welch 交叉功率谱密度（CSD）、加权相位滞后指数（wPLI）、虚部相干性（iCoh）、相位-幅度耦合（PAC）。
- **深度学习与神经大模型:** PyTorch、PEFT (LoRA/QLoRA)、因果大模型（Qwen2.5-7B, Llama-3）、序列到序列架构（BART-large）、残差矢量量化（EMA-RVQ）、Perceiver Resampler、多正例 InfoNCE 对比学习。
- **硬件与仪器开发:** 脑电模拟前端硬件设计（TI ADS1299 / ADS1292）、低功耗蓝牙（BLE）实时遥测、24GHz / 60GHz 毫米波雷达、DSP 滤波器算法、FPGA/Zynq 嵌入式系统。
- **算力与基础架构:** 拥有本地专属工作站 **NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB UMA)**、NVIDIA A100/H100 集群操作经验、Linux/Docker、混合精度训练 (BF16/FP16)。
- **语言能力:** **英语**（专业学术流利，熟练进行学术论文写作与国际会议英文报告）· **泰语**（母语）· **中文**（日常基础对话与听力交流）。
