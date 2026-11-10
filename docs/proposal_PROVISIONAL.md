# Project Proposal — PROVISIONAL DRAFT
*Not final: per assignment §4A, finalize only after the data inspection and an initial learned baseline have actually been run. No model results exist yet.*

**Question.** Does adding a soft Dice term to pixel-wise BCE (BCE + λ·Dice, λ=1) increase positive-slice Dice compared with BCE alone, without increasing the number of empty-mask slices receiving false-positive pixels, for an identical 2D U-Net, preprocessing, augmentation and training budget on the fixed patient-level split?

**Motivation (from metadata audit of train/val only).** Foreground is ~1.06% of training pixels; 65% of training slices (1,794/2,763) have empty masks; positive-slice lesion size is highly skewed (median 1,584 px; 10th pct 336 px; min 23 px; 21 slices <100 px). Plain BCE is dominated by background and may under-segment small lesions; Dice optimizes overlap directly but may encourage spurious positives on empty slices. Both are measurable here.

**Methods.** (i) Always-empty reference. (ii) Learned baseline: U-Net (from scratch) + BCE. (iii) Proposed: same + soft Dice. Training-only channel normalization; aligned flips/rot90; checkpoint and threshold selected on validation only; frozen before a single test evaluation.

**Comparisons.** Main: BCE vs BCE+Dice (3 seeds each). Diagnostic: (a) class-weighted BCE (pos_weight from train foreground ratio) tests whether any gain is just foreground reweighting; (b) Dice computed only on positive slices isolates whether Dice's effect on empty slices drives FP changes. Sensitivity: λ ∈ {0.5, 1, 2} and seed variation.

**Metrics.** Foreground Dice and IoU (pooled), positive-slice Dice/IoU, empty slices with any FP and FP pixel burden, patient-level Dice distribution; native 256×256; both-empty=1, one-empty=0. Subgroup: positive-slice Dice by lesion-size tertile (thresholds from train).

**Contradicting evidence.** Hypothesis rejected if BCE+Dice's positive-slice Dice gain is within seed variation, if weighted BCE matches it, or if empty-slice FP count rises beyond baseline seed range.

**Compute.** ~14 GPU runs on the user's AWS account (SageMaker), pending authentication and an approved cost cap.
