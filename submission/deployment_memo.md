Deployment Readiness Assessment for Facial Verification Fairness Audit System
===============================================================================

Prepared for: CTO, AI Governance Board, Responsible AI Review Committee
Date: 2026-05-28

1. Executive Summary
--------------------
Purpose
: The system embeds face images using a ResNet18 backbone into 128-dimensional vectors and performs thresholded cosine-similarity verification for biometric matching. The built pipeline is an artifact-driven audit: it computes ROC/DET curves, subgroup FAR/FRR, cross-group deltas, and exports structured audit artifacts.

Fairness Objective
: Evaluate demographic disparities across gender, age bins, and mapped skin-tone categories; quantify subgroup and cross-group disparities; test a mitigation strategy based on balanced subgroup sampling and weighted triplet loss.

Demographic Evaluation
: Groups defined by the FairFace-inspired schema (gender × age bins × skin-tone mapping). Subgroup and cross-group analyses are computed per pairwise combinations and aggregated into fairness summaries.

Mitigation Attempt
: A fairness-aware retraining run used balanced subgroup sampling and a weighted triplet loss. Artifacts: artifacts/best_mitigated_model.pth and results/mitigated_audit.json.

Recommendation (Bottom Line)
: Not recommended for high-stakes deployment in current state. Further validation, threshold calibration, larger balanced datasets, and additional mitigation are required prior to any production rollout.


2. System Overview
------------------
Architecture
- Embedding backbone: ResNet18 (pretrained initialization)
- Embedding dimension: 128
- Training objective: Triplet loss (weighted during mitigation)
- Verification metric: Cosine similarity between embeddings
- Decision rule: Fixed threshold (selected from validation, see results/threshold_analysis.json)
- Audit modality: subgroup and cross-group FAR/FRR analysis

Operational notes
- The pipeline is artifact-driven; key artifacts live under `results/` and `artifacts/`.


3. Dataset & Demographic Definitions
-----------------------------------
Data source
- Evaluation is performed on curated validation/audit pairs derived from a face dataset mapped to demographic bins (FairFace style). Raw dataset artifacts are not committed to the repository.

Demographic bins
- Gender groups: Male, Female
- Age bins: e.g., 0-19, 20-39, 40-59, 60+
- Skin-tone mapping: coarse mapping applied to match audit categories

Evaluation methodology
- Subgroup metrics computed per demographic slice (FAR, FRR, support counts)
- Cross-group metrics computed for pairwise group comparisons to identify intersectional instability
- Support thresholds enforced to warn on low-sample slices


4. Initial Audit Findings
-------------------------
(From results/initial_audit.json and results/fairness_summary.json)

Aggregate operating point (validation threshold)
- ROC AUC: ~0.53499 (results/threshold_analysis.json)
- Validation-selected threshold: ~0.3687
- Estimated operating FAR: ~0.3307, FRR: ~0.6077 (results/threshold_analysis.json)

Aggregate audit metrics (initial)
- Overall FAR (audit): ~0.324565
- Overall FRR (audit): ~0.628622

Subgroup disparities
- Worst FRR subgroup: `Male_0-19_Light` — FRR ≈ 0.717647 (results/fairness_summary.json)
- Largest FRR gap: ≈ 0.405147 between best and worst groups (results/fairness_summary.json)
- Most affected demographics: `Male_0-19_Light`, `Female_60+_Medium` (results/fairness_summary.json)

Cross-group issues
- Example: `Female_20-39_Dark__vs__Male_40-59_Dark` shows FAR=1.0 on audit pairs, indicating an unstable decision boundary for that pairing (results/analysis.json)


5. Bias Root Cause Analysis
---------------------------
(From results/analysis.json)

Data-level causes
- Representation imbalance: several demographic slices have small support (e.g., `Male_60+_Dark`). Small sample counts increase estimator variance and risk unstable fairness estimates.
- Sparse intersectional coverage magnifies cross-group errors; some pairings have support=1.

Model-level causes
- Embedding separability: the ResNet18-based embeddings do not uniformly separate identity manifolds across demographics.
- Threshold sensitivity: the selected global threshold produces large FRR variance across groups — suggests uncalibrated, non-linear score distributions across demographics.
- Pretraining mismatch: ImageNet-style pretraining may not supply equitable feature bases for demographic diversity in faces.


6. Mitigation Strategy
----------------------
Implemented steps
- Balanced subgroup sampling: training minibatches were drawn with subgroup balancing to increase representation during mitigation runs.
- Weighted triplet loss: loss weighting emphasized underrepresented slices to encourage more uniform margins.
- Fairness-aware retraining: retrain on balanced subsets and validate on held-out audit pairs.

Rationale
- These are targeted, pragmatic interventions to reduce FRR disparities by encouraging the model to allocate representational capacity to under-served groups.

Limits
- These methods can change score distributions; without threshold recalibration they can trade-away FAR for lower FRR.


7. Post-Mitigation Evaluation
-----------------------------
(From results/mitigated_audit.json, results/fairness_comparison.json, results/overall_metrics.json)

Aggregate changes
- Initial accuracy: ~0.5234 → Mitigated accuracy: ~0.5060 (results/overall_metrics.json)
- Initial FAR: ~0.3246 → Mitigated FAR: ~0.5194 (increase)
- Initial FRR: ~0.6286 → Mitigated FRR: ~0.4686 (improvement)

Trade-off analysis
- The mitigation decreased FRR (reduced false rejections) at the expense of a meaningful FAR increase. This reflects a classic accessibility-vs-security trade-off: fewer legitimate users are rejected, but more impostors may be incorrectly accepted.
- Fairness comparison shows average disparity reduction ≈ 0.1047 (results/fairness_comparison.json), meaning some subgroup gaps narrowed; however, several cross-group pairings worsened.

Interpretation
- Mitigation partially succeeded at reducing rejection-based harms but introduced acceptance-based harms (FAR) that may be unacceptable in authentication contexts.
- Net deployment utility depends heavily on context: physical access vs. convenience vs. fraud risk.


8. Ethical & Operational Risks
-----------------------------
- Disparate impact: High FRR for specific subgroups leads to unequal denial of service.
- False rejection harms: Denying legitimate users can cause service exclusion, reputational harm, and safety risks.
- False acceptance harms: Elevated FAR raises security and safety risks (e.g., unauthorized access).
- Biometric discrimination risk: Systemic biases can create differential treatment; regulators may regard biometric systems as high-risk.
- Threshold selection: A single global threshold amplifies demographic imbalance; using a global threshold without calibration risks unequal treatment.
- Audit limitations: Sample size and dataset representativeness constrain the generalizability of these findings.


9. Final Deployment Recommendation
---------------------------------
- Do NOT deploy this model in high-stakes or sensitive contexts in its current form.
- Required preconditions for deployment:
  - Recalibrate thresholds using stratified/ subgroup-aware calibration methods.
  - Collect larger, balanced datasets across underrepresented slices and intersections.
  - Apply stronger debiasing strategies, including adversarial or domain-adaptive approaches.
  - Implement continuous fairness monitoring and an incident response playbook.
  - Evaluate operational tolerance for FAR vs FRR trade-offs with business and legal stakeholders.


10. Future Work
---------------
Recommended technical directions
- Expand and rebalance validation and training datasets to reduce estimator variance.
- Investigate adversarial fairness learning and domain adaptation to reduce embedding bias.
- Explore subgroup-aware calibration (per-group thresholds or score normalization) and adaptive thresholding under operational constraints.
- Perform longitudinal monitoring with data drift detection and periodic re-audits.
- Consider human-in-the-loop fallbacks for high-risk decisions.

Recommended governance
- Run external audits and pre-deployment impact assessments.
- Policy alignment: ensure compliance with biometric-specific regulations and fairness requirements.


Appendix: Key Artifact References
---------------------------------
- results/initial_audit.json
- results/mitigated_audit.json
- results/fairness_summary.json
- results/fairness_comparison.json
- results/overall_metrics.json
- results/analysis.json
- results/threshold_analysis.json


