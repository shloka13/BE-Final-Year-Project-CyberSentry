# CyberSentry — Secondary Review Component Evaluation Report

## 1. Overview and Objective

The **Secondary Review** component in CyberSentry acts as a post-classification safety layer within the `InvestigationAgent`. Its objective is to identify network flows exhibiting suspicious structural feature combinations that warrant manual review by a human security analyst, even when the primary multi-class ML classifier (XGBoost) predicts `Benign` with high confidence and the unsupervised anomaly detector (Isolation Forest) does not flag the flow.

Crucially:
- Secondary review does **not** alter the original ML classifier's prediction, confidence, or anomaly scores.
- Triggering secondary review is **not** treated as confirmed proof of malicious or Botnet activity; it is explicitly documented as a heuristic screening indicator recommending analyst investigation.
- It provides defense-in-depth against blind spots caused by training data distribution shifts.

---

## 2. Methodology and Rule Configuration

### Heuristic Rule: `zero_fwd_minimal_packet_size`
The active heuristic evaluates the following observed network traffic pattern:
1. `Subflow Fwd Bytes == 0`: Zero payload sent in the forward direction.
2. `Bwd Packet Length Min <= 6`: Minimal backward packet size (consistent with TCP control or header-only frames).
3. `Max Packet Length <= 6`: The maximum observed packet across the entire flow is extremely small (<= 6 bytes).
4. `target_ports`: Optional port constraint (default: `None`, applies across all ports).

### Active Configuration
- **Rule Name**: `zero_fwd_minimal_packet_size`
- **Max Forward Bytes**: `0.0`
- **Max Backward Packet Length Min**: `6.0`
- **Max Overall Packet Length**: `6.0`
- **Port Filter**: `None` (Active default across all ports)

### Reproduction Command
```bash
python -m src.ml.evaluate_secondary_review
```
Alternatively, for split-specific evaluations or port filtering:
```bash
python -m src.ml.evaluate_secondary_review --split val
python -m src.ml.evaluate_secondary_review --split test
python -m src.ml.evaluate_secondary_review --port-filter 8080
```

---

## 3. Dataset Splits & Feature Semantics

The evaluation was executed on the processed CIC-IDS2017 splits stored under `data/processed/`:
- **Validation Split**: `data/processed/val.csv.gz` (309,496 flows)
- **Held-out Test Split**: `data/processed/test.csv.gz` (309,373 flows)

All required heuristic features (`Subflow Fwd Bytes`, `Bwd Packet Length Min`, `Max Packet Length`) were 100% present (0 missing / unusable values in both splits).

> **Disclosure on Held-Out Test Set**:
> The test set evaluation reported here was performed using the fixed heuristic parameters developed and calibrated on validation data. However, as documented in earlier project audit reports, the test set labels and feature distributions were previously inspected during preliminary Botnet generalization failure diagnostics. Therefore, it is disclosed that the test set is not an untouched blind evaluation set.

---

## 4. Evaluation Results

### Summary Table

| Metric | Validation Split (`val`) | Held-out Test Split (`test`) |
| :--- | :---: | :---: |
| **Total Flows** | 309,496 | 309,373 |
| **Total Flagged for Review** | 42,742 (13.81%) | 38,244 (12.36%) |
| **Botnet Flows Flagged** | **198 / 292 (67.81%)** | **293 / 293 (100.00%)** |
| **Benign Flows Flagged (FPR)** | 39,832 / 279,482 (14.25%) | 35,473 / 279,096 (12.71%) |
| **Missing / Unusable Features** | 0 | 0 |

### Per-Class Flagging Rates

#### Validation Split (`val`)
| Class | Total Flows | Flagged Flows | Flagging Rate |
| :--- | :---: | :---: | :---: |
| **Benign** | 279,482 | 39,832 | 14.25% |
| **Botnet** | 292 | 198 | **67.81%** |
| **BruteForce** | 1,819 | 72 | 3.96% |
| **DoS** | 27,577 | 2,327 | 8.44% |
| **Infiltration** | 5 | 0 | 0.00% |
| **WebAttack** | 321 | 313 | 97.51% |

#### Held-out Test Split (`test`)
| Class | Total Flows | Flagged Flows | Flagging Rate |
| :--- | :---: | :---: | :---: |
| **Benign** | 279,096 | 35,473 | 12.71% |
| **Botnet** | 293 | 293 | **100.00%** |
| **BruteForce** | 772 | 36 | 4.66% |
| **DoS** | 28,884 | 2,143 | 7.42% |
| **Infiltration** | 6 | 0 | 0.00% |
| **WebAttack** | 322 | 299 | 92.86% |

---

## 5. Key Findings and Analysis

1. **Botnet Escalation Recovery**:
   - In the baseline system, the XGBoost classifier classified **0 out of 293 (0.0%)** Botnet test flows because all Botnet test flows belonged to an unrepresented Subtype B (one-directional C2 / heartbeat polling with zero forward payload).
   - The secondary-review heuristic flags **293 out of 293 (100.0%)** Botnet test flows, successfully escalating every missed Botnet test flow for human analyst investigation.
   - On the validation set, the heuristic flags **198 out of 292 (67.81%)** Botnet flows, representing **100% of the Subtype B Botnet flows** present in validation. The remaining 94 validation Botnet flows are Subtype A (bidirectional flows with median forward bytes ~206).

2. **WebAttack Flagging**:
   - A large proportion of `WebAttack` flows (97.51% in validation, 92.86% in test) match the zero-forward/small-packet pattern (predominantly truncated HTTP scans and teardowns). The secondary review heuristic acts as an effective triage net for these as well.

3. **Benign False-Positive Rate**:
   - Across all destination ports, the benign flagging rate is **12.71%** on test (14.25% on validation). These benign flows consist of normal TCP reset/teardown events (e.g., ports 80 and 443) that exchange zero forward data and small header packets.
   - When configured with a port-restricted rule (e.g., `target_ports=[8080]`), the benign FPR drops to **0.00036%** (1 in 279,482) while still capturing 100% of the Subtype B Botnet flows.

---

## 6. Known Limitations

1. **Not an ML Model Replacement**: The secondary-review heuristic is a rule-based triage safety net, not an autonomous classifier. It identifies conditions under which model blind spots are known to occur.
2. **Generic Zero-Forward Traffic Overhead**: When evaluated unconstrained across all ports, the ~13% benign review rate requires analyst triage bandwidth. In production environments, pairing the heuristic with host reputation or port filtering significantly reduces false-positive triage volume.
3. **Subtype A Invariance**: The current heuristic specifically targets one-directional / zero-forward traffic (Subtype B). Multi-packet bidirectional Botnet traffic (Subtype A) relies on primary ML classification and Isolation Forest.
