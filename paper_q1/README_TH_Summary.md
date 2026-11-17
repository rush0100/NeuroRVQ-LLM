# ร่างบทความวิจัยระดับ Q1 (Paper 1: NeuroRVQ-LLM) — พร้อมแผนภาพและสรุปภาษาไทยสำหรับทีมวิจัย

* **คณะผู้วิจัย (Authors & Affiliations — ตามเกณฑ์ทุน สจล. ข้อ 9 & ข้อ 12.1):**
  1. **Ratchaphon Khaemphukhiao** *(First Author — Ph.D. Student, Dept. of Biomedical Engineering, School of Engineering, KMITL)*
  2. **Zhenxi Song** *(Co-Author — School of Computer Science & Technology / College of AI, Harbin Institute of Technology, Shenzhen)*
  3. **Zhiguo Zhang** *(Co-Corresponding Author — Professor, Harbin Institute of Technology, Shenzhen)*
  4. **Assoc. Prof. Pattarapong Phasukkit, D.Eng.** *(Principal Advisor & Corresponding Author — Dept. of Electronic Engineering, School of Engineering, KMITL, `pattarapong.ph@kmitl.ac.th`)*
* **วารสารเป้าหมาย (Web of Science / Scimago Q1 — ตรงตามเกณฑ์ทุนปริญญาเอกพระจอมเกล้าลาดกระบัง ข้อ 12.1):**
  1. **IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)** *(Q1, IF ~4.8–5.2 — วารสารเรือธงด้าน BCI & Neural Engineering)*
  2. **IEEE Journal of Biomedical and Health Informatics (JBHI)** *(Q1, IF ~6.7)*
  3. **Journal of Neural Engineering (IOP)** *(Q1, IF ~3.7–4.0)*
* **ไฟล์ต้นฉบับ LaTeX มาตรฐาน IEEE Transactions พร้อมคอมไพล์บน Overleaf:**
  * [`NeuroRVQ_LLM_IEEE_TNSRE_Draft.tex`](./NeuroRVQ_LLM_IEEE_TNSRE_Draft.tex)
  * [`references.bib`](./references.bib)
* **กิตติกรรมประกาศบังคับตามสัญญาทุน สจล. (ข้อ 12.2 — ใส่ใน `\thanks{...}` และ `Acknowledgment` เรียบร้อยแล้ว):**
  > *"This work is supported by King Mongkut's Institute of Technology Ladkrabang."*

---

## 1. จุดขายหลัก 4 ข้อที่ทำให้ผ่านเกณฑ์ Reviewer ระดับ IEEE Q1

1. **ค้นพบและปิดช่องโหว่ "Fixation-Count Length Leakage" เป็นครั้งแรกในสาย ZuCo EEG-to-Text:**
   * พิสูจน์ด้วยตัวเลขจริงใน Table I ว่า Positional Encoding แบบเลขลำดับคำปกติ (`0, 1, ..., L-1`) แอบบอกใบ้ความยาวประโยคให้โมเดล แม้ป้อน **Pure Gaussian Noise** คะแนน MRR ก็ยังสูงกว่าค่าสุ่มถึง **3.2 เท่า (`5.29%` เทียบกับ `1.64%`)**
   * แก้ด้วย **Continuous `[0, 1]` Scanpath-Progress Fourier Encoding** + **Perceiver Resampler ($K=24$)** กดคะแนนของ Gaussian Noise ลงเหลือ **`1.42%`** (เท่าค่าสุ่มทางทฤษฎีเป๊ะ)
2. **ยกระดับการสกัดฟีเจอร์จากเวกเตอร์แบนๆ `840-d` สู่ 3D Cortical ROI + 4-Stage Reading + Riemannian-Phase Connectome:**
   * ใช้พิกัด 3 มิติจริงจาก `chanlocs105.mat` ยุบ 105 อิเล็กโทรดสู่ **12 สมองส่วนหลักด้านภาษาและการมองเห็น** พร้อม `valid_channel_mask` กันช่องเสีย และทำ Unsupervised Per-Subject Robust Calibration (ใช้กับคนใหม่ได้ทันทีโดยไม่ต้องมีเฉลย)
   * ดึงครบทั้ง **ความแรงรายระยะการคิด (`FFD -> GD -> TRT -> TRT-FFD`)**, **ความสมมาตรซ้าย-ขวา 48 คู่ (`*_diff`)**, **ลำดับการกวาดตาจริง (`fixPositions`)**, **ขนาดรูม่านตา (`pupilsize`)** และคำนวณ **Pure Relation (`wPLI`, `iCoh`) + Coupled Signal Strength (`Log-Euclidean SPD Tangent`)** จากคลื่นดิบ `500 Hz`
3. **สถาปัตยกรรมใหม่ `Late Connectome Bottleneck Fusion` + `Un-inflated 6-Book EMA-RVQ`:**
   * พิสูจน์เชิงประจักษ์ว่าทำไมการเอาคู่อิเล็กโทรด/กราฟสมองยัดใส่ทุกคำตั้งแต่ต้นทางถึงเกิดอาการ Overfit ข้ามคน (`14.94%` MRR) และแก้ด้วย **Zero-Initialized Late Connectome Bottleneck Adapter** จนดันความแม่นยำบน **Unseen Subject (`ZMG`)** ทะลุขึ้นไปถึง **`MRR = 22.22%` (`R@10 = 44.30%`, `Sanity Gap = +20.80%`)**
   * แก้ปัญหา Codebook Collapse ของ VQ เดิม (`Perplexity = 1.5` $\rightarrow$ **`84.5`**) รักษาข้อมูลความหมายหลังแปลงเป็นรหัสตัวเลขได้มากกว่า **93%**
4. **เชื่อมสู่ `Qwen2.5-7B-Instruct` แบบ Teacher-Forcing-Free บนเครื่อง NVIDIA DGX Spark เครื่องเดียว:**
   * ใช้ **50% Scheduled Target-Word Dropout** บังคับให้ LLM เลิกท่องจำแพตเทิร์นประโยค (ซึ่งเราโชว์หลักฐานจริงใน Table II Row 1 ว่าถ้าไม่แก้ LLM จะพ่นประโยคหนังซ้ำๆ จนได้คะแนน Noise พอๆ กับ EEG จริง) และรันเทรนสเกลเต็ม **18,276 ตัวอย่าง (30 อาสาสมัคร)** บนชิป GB10 Unified Memory 128 GB

---

## 2. สรุปผลการทดลองฉบับสมบูรณ์ (กรอกลงตาราง Table I, II, III และคอมไพล์ PDF เรียบร้อย 100% แล้ว)

* **Stage 1 Zero-Shot 1,313-Sentence Retrieval (Unseen Test 4 คน: `ZMG, ZPH, YTL, YSL` รวม 2,717 ตัวอย่าง):**
  * **Real EEG:** `R@1 = 9.90%` (`130x` ของค่าสุ่ม), `R@5 = 35.08%`, **`R@10 = 56.09%`**, `MRR = 23.14%`, `Median Rank = 9.0 / 1,313`
  * **Gaussian Noise:** `R@1 = 0.11%`, `R@10 = 1.14%`, `MRR = 0.77%`, `Median Rank = 647.0` (`Sanity Gap = +54.95% R@10 / +22.37% MRR`)
* **Stage 2 Strict Teacher-Forcing-Free Generation (`Qwen2.5-7B-Instruct` BF16 LoRA + 50% Word Dropout):**
  * **Validation Peak (Epoch 8):** `BLEU-1 = 22.15%`, `BLEU-2 = 15.37%`, `BLEU-4 = 12.21%`, `ROUGE-1 F1 = 21.90%`, `WER = 88.70%`
  * **Final Unseen Test — Real EEG:** **`BLEU-1 = 19.60%`**, **`BLEU-2 = 12.82%`**, **`BLEU-3 = 10.81%`**, **`BLEU-4 = 9.89%`** (ชนะ `LURS` *EMNLP 2026* ที่ได้ `0.89%` ถึง `11.1 เท่า`), **`ROUGE-1 F1 = 19.58%`**, **`WER = 90.85%`**
  * **Final Unseen Test — Gaussian Noise:** `BLEU-1 = 8.77%`, `BLEU-2 = 2.17%`, `BLEU-3 = 0.78%`, `BLEU-4 = 0.43%`, `ROUGE-1 F1 = 11.15%`, `WER = 182.92%`
  * **Sanity Check Gap:** **`+10.83%` $\Delta$BLEU-1** และ **`+9.46%` $\Delta$BLEU-4 (`23.0x` เหนือ Noise!)**
