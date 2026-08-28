# VoxPulse Voice Attribute Dataset Audit & Hygiene Report

**Generated via**: `scripts/inspect_age_dataset.py` & `scripts/audit_dataset.py`  
**Dataset Scale**: 3,458 valid audio recordings across 3,028 unique genuine speakers  
**Speaker Partitioning**: Strict speaker-isolated splits ($\text{Train} \cap \text{Val} = \emptyset, \; \text{Train} \cap \text{Test} = \emptyset, \; \text{Val} \cap \text{Test} = \emptyset$)

---

## 1. Overall Dataset Summary

| Metric | Total Counts | Notes |
| :--- | :---: | :--- |
| **Total Audio Files** | 3,458 | All 16kHz mono WAV |
| **Unique Genuine Speakers** | 3,028 | 0 overlapping speakers between splits |
| **Age-Labeled Recordings** | 2,910 | Real self-reported demographic metadata |
| **Gender-Labeled Recordings** | 3,156 | Real self-reported demographic metadata |
| **Average Audio Duration** | 3.82 seconds | Range: 1.0s to 10.0s |
| **Dataset Sources** | 4 corpora | Global Voices (`globe_v2`), Common Voice Age & Gender, FLEURS |

---

## 2. Split-by-Split Breakdown

### 2.1 Training Split (70% — 2,120 Speakers | 2,418 Samples)

| Age Bracket | Audio Samples | Unique Speakers | Male / Female / Unk | Mean Duration | Median Duration | Min - Max Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **18-30** | 1,111 | 1,065 | 745 / 304 / 0 | 3.51 s | 3.32 s | 1.0 s - 9.6 s |
| **31-45** | 693 | 676 | 576 / 54 / 0 | 4.10 s | 3.84 s | 1.1 s - 9.9 s |
| **46-60** | 136 | 136 | 56 / 60 / 0 | 4.51 s | 4.12 s | 1.1 s - 9.3 s |
| **60+** | 102 | 102 | 28 / 0 / 0 | 6.22 s | 5.91 s | 1.3 s - 10.0 s |
| *Gender-Only / Background* | 376 | 243 | 210 / 166 / 0 | 3.45 s | 3.20 s | 1.0 s - 8.5 s |
| **Total Train** | **2,418** | **2,120** | **1,615 / 584 / 0** | **3.80 s** | **3.55 s** | **1.0 s - 10.0 s** |

### 2.2 Validation Split (15% — 454 Speakers | 520 Samples)

| Age Bracket | Audio Samples | Unique Speakers | Male / Female / Unk | Mean Duration | Median Duration | Min - Max Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **18-30** | 219 | 212 | 157 / 54 / 0 | 3.38 s | 3.20 s | 1.0 s - 7.3 s |
| **31-45** | 154 | 153 | 126 / 14 / 0 | 4.12 s | 3.90 s | 1.1 s - 9.9 s |
| **46-60** | 33 | 33 | 10 / 20 / 0 | 4.07 s | 3.75 s | 1.3 s - 7.4 s |
| **60+** | 15 | 15 | 5 / 0 / 0 | 4.88 s | 4.50 s | 2.0 s - 10.0 s |
| *Gender-Only / Background* | 99 | 71 | 51 / 48 / 0 | 3.52 s | 3.30 s | 1.0 s - 8.1 s |
| **Total Validation** | **520** | **454** | **349 / 136 / 0** | **3.72 s** | **3.50 s** | **1.0 s - 10.0 s** |

### 2.3 Held-Out Test Split (15% — 454 Speakers | 520 Samples)

| Age Bracket | Audio Samples | Unique Speakers | Male / Female / Unk | Mean Duration | Median Duration | Min - Max Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **18-30** | 219 | 208 | 135 / 71 / 0 | 3.70 s | 3.50 s | 1.2 s - 9.1 s |
| **31-45** | 181 | 175 | 156 / 11 / 0 | 4.15 s | 3.95 s | 1.1 s - 9.0 s |
| **46-60** | 29 | 29 | 10 / 10 / 0 | 5.15 s | 4.80 s | 1.4 s - 8.5 s |
| **60+** | 18 | 18 | 6 / 0 / 0 | 5.19 s | 4.90 s | 1.5 s - 9.0 s |
| *Gender-Only / Background* | 73 | 54 | 30 / 43 / 0 | 3.65 s | 3.40 s | 1.0 s - 8.2 s |
| **Total Test** | **520** | **454** | **337 / 135 / 0** | **3.95 s** | **3.70 s** | **1.0 s - 9.1 s** |

---

## 3. Speaker Isolation Audit

Programmatically verified via set intersection assertions:

$$\begin{aligned}
|\text{Train Speakers} \cap \text{Validation Speakers}| &= 0 \quad \text{[PASSED]} \\
|\text{Train Speakers} \cap \text{Test Speakers}| &= 0 \quad \text{[PASSED]} \\
|\text{Validation Speakers} \cap \text{Test Speakers}| &= 0 \quad \text{[PASSED]}
\end{aligned}$$

---

## 4. Demographic Confusion-Risk Analysis

1. **Younger Skew ($18\text{-}30$ vs $31\text{-}45$)**:
   - $18\text{-}30$ represents $54.4\%$ and $31\text{-}45$ represents $33.9\%$ of training data.
   - Combined, these two brackets account for $88.3\%$ of the real-world dataset.
   - **Risk**: Standard unweighted classifiers will predict $18\text{-}30$ for almost all inputs.
   - **Mitigation**: We enforce inverse class weighting ($w_{46-60}=3.754, w_{60+}=5.005$) and train ordinal cumulative binary threshold heads ($>30, >45, >60$) rather than single unconstrained multi-class softmax.

2. **Older Speaker Under-representation ($60+$)**:
   - Only 102 training speakers and 18 test speakers exist in the $60+$ bracket across open public corpora.
   - **Risk**: A pure linear classifier trained on 102 samples can easily overfit to individual vocal characteristics rather than general age indicators (such as lower vocal tract resonance and reduced fundamental frequency variability).
   - **Mitigation**: The ordinal model structure guarantees that a $60+$ prediction requires passing all three progressive thresholds ($>30 \land >45 \land >60$), constraining predictions monotonically.
