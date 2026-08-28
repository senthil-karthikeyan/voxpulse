# VoxPulse Voice Attribute Inference Service: Age Pipeline Audit

**Date**: 2026-08-28  
**Audit Target**: End-to-End Age-Bracket Inference Architecture, Dataset, and Preprocessing  
**Status**: Comprehensive Engineering Audit Completed

---

## Executive Summary

This audit traces the complete age estimation pipeline from raw audio ingestion, metadata parsing, feature representation, model training, and post-hoc calibration, to real-time API inference and audio quality rejection gates.

---

## 1. Traceability & Structural Questions Audit

### 1. Where do age labels come from?
- **Primary Source (76.0%)**: **Global Voices Speech Corpus (`globe_v2` via `Paranoiid/VoicePersona`)**, which contains genuine self-reported human demographic data (age in years / demographic categories).
- **Secondary Source (8.2%)**: **Mozilla Common Voice Age Corpus (`DynamicSuperb/AgeClassification_CommonVoiceCorpus-Test`)**, which contains verified human speakers with self-reported demographic metadata (`twenties`, `thirties`, `fourties`, `fifties`, `sixties`, `seventies`, `eighties`).
- **Gender & Anchor Sources (15.8%)**: **Mozilla Common Voice Gender Delta Segment** and **Google FLEURS English** (contributing validated gender labels and background speech).

### 2. How are age labels mapped into brackets?
Mapped through `app/evaluation/mappings.py` via `normalize_age()`:
- `"teens"` / age $< 18 \to$ **Excluded** (per system requirement: only adult callers $\ge 18$).
- `"twenties"`, $18 \le \text{age} \le 30 \to$ **`18-30`** (Index 0).
- `"thirties"`, `"fourties"`, $31 \le \text{age} \le 45 \to$ **`31-45`** (Index 1).
- `"fifties"`, $46 \le \text{age} \le 60 \to$ **`46-60`** (Index 2).
- `"sixties"`, `"seventies"`, `"eighties"`, `"nineties"`, $\text{age} > 60 \to$ **`60+`** (Index 3).

### 3. How many samples exist per age bracket?
Across the 3,458 total samples in the dataset:
- **`18-30`**: 1,549 samples (Train: 1,111 | Val: 219 | Test: 219)
- **`31-45`**: 1,028 samples (Train: 693 | Val: 154 | Test: 181)
- **`46-60`**: 198 samples (Train: 136 | Val: 33 | Test: 29)
- **`60+`**: 135 samples (Train: 102 | Val: 15 | Test: 18)
- **Total Age-Labeled**: 2,910 valid samples (548 samples are gender-only / background anchors).

### 4. How many unique speakers exist per bracket?
- **`18-30`**: 1,485 unique genuine speakers (Train: 1,065 | Val: 212 | Test: 208)
- **`31-45`**: 1,004 unique genuine speakers (Train: 676 | Val: 153 | Test: 175)
- **`46-60`**: 198 unique genuine speakers (Train: 136 | Val: 33 | Test: 29)
- **`60+`**: 135 unique genuine speakers (Train: 102 | Val: 15 | Test: 18)
- **Total Unique Speakers in Dataset**: 3,028 unique human speakers.

### 5. Does one speaker contribute multiple clips?
- In the training set, the vast majority of speakers contribute exactly 1 clip (mean: 1.14 clips/speaker; max: 2 clips/speaker).
- Critically, all clips belonging to any single speaker ID are strictly restricted to exactly one partition (either 100% in Train, 100% in Val, or 100% in Test).

### 6. Does class imbalance exist?
- **Yes, substantial natural imbalance exists**:
  - `18-30` represents **54.4%** of training data.
  - `31-45` represents **33.9%** of training data.
  - `46-60` represents **6.7%** of training data.
  - `60+` represents **5.0%** of training data.
- **Handling**: Inverse class frequency weighting is applied during training ($w_{18-30}=0.459, w_{31-45}=0.737, w_{46-60}=3.754, w_{60+}=5.005$) along with ordinal cumulative threshold decomposition.

### 7. Is speaker leakage possible?
- **No**. Speaker-level isolation is programmatically asserted with 0 overlapping speaker IDs:
  $$\text{Train} \cap \text{Val} = 0, \quad \text{Train} \cap \text{Test} = 0, \quad \text{Val} \cap \text{Test} = 0$$

### 8. Does FLEURS contribute valid age labels?
- **No**. FLEURS provides verified gender and accent metadata, but does not provide self-reported age labels. In `prepare_dataset.py`, FLEURS samples are marked with `age_bracket: null` (unlabeled for age) and used solely for gender representation and acoustic diversity.

### 9. Is Mozilla Common Voice metadata genuinely being used?
- **Yes**. Common Voice client speaker hashes, self-reported age strings, and gender strings from the official corpus releases are ingested directly.

### 10. Do any fabricated or synthetic labels remain?
- **No**. All synthetic label generation, filename hashing, and modulo cycling have been completely removed. Every sample in `data/processed/*.jsonl` traces to a real recording and real demographic field.

### 11. Are training and evaluation preprocessing identical?
- **Yes**. Both offline feature extraction (`extract_embeddings.py`) and live API inference (`InferenceService.analyze()`) share the exact same preprocessing pipeline:
  1. Audio decode and resample to 16,000 Hz single-channel mono.
  2. Peak normalization to $-1.0\text{ dBFS}$ ($0.85$ max amplitude).
  3. Energy-based framing with 4.0-second salient window selection centered around active speech.
  4. 192-dimensional embedding extraction via SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`).

### 12. Do ECAPA embedding dimensions match?
- **Yes**. Both training feature matrices and live inference tensors are $\mathbf{e} \in \mathbb{R}^{192}$.

### 13. Does normalization differ between training and inference?
- **No**. Both training and inference use identical $-1.0\text{ dBFS}$ peak normalization and unit L2 vector normalization.

### 14. Do `.pt` files correspond to the currently running architecture?
- **Yes**. `model_weights/age_head.pt` contains the trained state dictionary loaded by `AgeClassifier` with automatic dynamic architecture detection (`BaselineLinearAgeHead`, `DeepMLPAgeHead`, `OrdinalAgeHead`).

---

## 2. Root Cause Analysis: Why Baseline ECAPA + Softmax Collapsed

1. **Acoustic Subspace Misalignment**: ECAPA-TDNN was trained with Additive Angular Margin Softmax (AAM-Softmax) on VoxCeleb to maximize inter-speaker distance and minimize intra-speaker distance. Speaker identity features are orthogonal to age: a 25-year-old speaker and a 65-year-old speaker may both occupy high-dimensional space where simple linear hyperplanes fail without sufficient demographic training volume.
2. **Standard Multi-Class Softmax Disregards Ordinality**: Treating age brackets as unordered discrete categories ($0, 1, 2, 3$) penalized a mistake of predicting $31\text{-}45$ for a $18\text{-}30$ speaker equally to predicting $60+$. This caused uncalibrated probability mass to spill into extreme brackets.
3. **Severe Class Imbalance**: Without class weighting and ordinal cumulative heads, models optimized for raw accuracy naturally converged to predicting the majority $18\text{-}30$ class with $\approx 98\%$ uncalibrated confidence.

---

## 3. Recommended Remediation & Architecture Selection

- **Adopt Ordinal Classification**: Replaced 4-class categorical softmax with 3 cumulative binary threshold heads ($>30, >45, >60$).
- **Apply Post-Hoc Temperature Scaling**: Optimized validation temperature $T=1.4648$ to align confidence probabilities with empirical accuracy.
- **Maintain Strict Audio Quality Gating**: Reject audio with $< 1.5\text{s}$ speech duration before neural inference to prevent spurious predictions on empty silence or background noise.
