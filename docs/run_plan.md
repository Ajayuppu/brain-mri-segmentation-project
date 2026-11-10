# GPU run plan (to price; not yet run)
Fixed budget per run: U-Net base 32, 50 epochs, batch 16, AdamW 1e-3 cosine, flips/rot90, val-selected checkpoint (pooled fg Dice) and threshold (grid 0.1–0.9).

| Block | Configs | Seeds | Runs |
|---|---|---|---|
| Learned baseline (BCE) | bce_baseline | 0,1,2 | 3 |
| Proposed (BCE+Dice, λ=1) | bce_dice | 0,1,2 | 3 |
| Diagnostic A: simpler reweighting | weighted_bce (pos_weight from train fg ratio, capped 100) | 0,1,2 | 3 |
| Diagnostic B: Dice on positive slices only | bce_dice_posonly | 0,1,2 | 3 |
| Sensitivity: Dice weight | bce_dice_w0.5, bce_dice_w2 | 0 | 2 |
| Optional pilot (initial baseline for proposal) | bce_baseline short | 0 | 1 |
| **Total** | | | **14 (+1 pilot)** |

Always-empty reference: CPU, free. Test: one evaluation after freeze.
Rough sizing (UNVERIFIED estimate): 2,763 slices × 50 epochs on one T4/A10G ≈ 10–30 min/run → ~3–7 GPU-hours total. Price on SageMaker ml.g4dn.xlarge or ml.g5.xlarge (spot optional) only after AWS auth + user cost cap. Measure pilot runtime first and re-estimate.
