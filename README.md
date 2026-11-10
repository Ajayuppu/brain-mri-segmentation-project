# LGG Brain MRI Binary Segmentation — BCE vs BCE+soft-Dice investigation

Status: **code package prepared; no full training run yet.** All metrics in this repo are from smoke tests unless a run_meta.json says `"smoke_test": false`.

## Layout
- `src/lggseg/` data (manifest loader, tifffile channel-preserving read, binary masks, train-only channel stats, aligned flip/rot90 aug), `model.py` (2D U-Net from scratch, ~7.76M params at base=32), `losses.py`, `metrics.py`.
- `scripts/audit_metadata.py` split/hash/patient-disjoint checks; train/val label stats only.
- `scripts/eval_reference.py` always-empty reference on val.
- `scripts/train.py` trains one config; checkpoint and threshold chosen **on val only**; writes `run_meta.json`, `train_log.json`, `best.pt`, `channel_stats.json`.
- `scripts/freeze.py` one-shot freeze → `FROZEN.json` + `FROZEN.json.sha256`. Hashes per run `best.pt`, `channel_stats.json` (train-only normalization; also embedded verbatim), `run_meta.json` (threshold, config), and eval-path source (`src/lggseg/{data,model,metrics,protocol}.py`, `scripts/evaluate_test.py`). Refuses smoke runs and stats not `fitted_on: train`.
- `scripts/evaluate_test.py` (logic in `src/lggseg/protocol.py`): re-verifies every hash, then atomically claims the test set (`TEST_EVAL.lock`, O_EXCL) and an append-only `TEST_EVAL_LEDGER.json` **before** reading test data. Completed → never re-runs (deleting the lock alone does not reopen). Failure is logged; retry only with `--retry-reason` and only if it failed before test labels were loaded. Hard crash (status `started`) → refused, manual disclosure required.
- `scripts/predict.py` inference to 256×256 binary PNG masks.
- `configs/` identical budgets; only loss differs.
- `tests/` metric conventions (both empty=1, exactly one empty=0), alignment, shapes, losses.

## Reproduce
```bash
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
pytest -q tests
# DATA = local copy of Drive folder LGG-MRI-Segmentation/dataset (manifests + train/ val/ test/)
python scripts/audit_metadata.py --data-root $DATA
python scripts/eval_reference.py --data-root $DATA
for c in bce_baseline bce_dice; do for s in 0 1 2; do python scripts/train.py --config configs/$c.yaml --data-root $DATA --seed $s; done; done
# ...diagnostic + sensitivity (docs/run_plan.md); then, ONLY after method decisions are final:
python scripts/freeze.py --runs runs/<chosen>...
python scripts/evaluate_test.py --data-root $DATA --i-confirm-method-is-frozen
```

## Test-protocol guarantees and limits
Evaluation is two-pass: PASS 1 predicts from test **images only** (no mask path opened); the ledger then records `test_labels_loaded=true`; PASS 2 reads masks and scores. Fix 2026-11-08 03:2x EST: earlier code used `SegDataset` (reads masks) during prediction, so a failure there could wrongly permit a "pre-label" retry. Regression tests instrument mask reads/file opens (fail on old code, pass now). Residual limit: the ledger marks label access at the start of PASS 2, conservatively — a failure after the mark but before any mask byte is read still blocks retry.

Guaranteed by code: tampering with frozen checkpoints, normalization stats, run metadata/thresholds, the freeze record, or hashed eval code is detected and evaluation refuses; a second evaluation via this code path is refused. **Not** guaranteed: someone can edit the code, delete the ledger/out dir, or score test predictions with another script. The ledger, freeze hashes and the Google Doc log are disclosed in the report as evidence; honesty of the experimenter remains the final safeguard.

## Metric conventions
All at native 256×256. Per-slice Dice/IoU: both empty = 1, exactly one empty = 0. Reported: pooled foreground Dice/IoU, mean slice Dice/IoU, positive-slice Dice/IoU, empty slices with any FP + FP pixel burden, patient-level Dice (pooled per patient) mean/median/std/min. Pixel accuracy not used.

## Attribution
- Data: Buda et al., Comput. Biol. Med. 109 (2019) 218–225; TCIA TCGA-LGG; Kaggle mateuszbuda/lgg-mri-segmentation; CC BY-NC-SA 4.0 (noncommercial; course redistribution reorganized files only).
- U-Net architecture: Ronneberger et al., MICCAI 2015 (implemented here from scratch; no pretrained weights).
- Soft Dice loss: Milletari et al., V-Net, 3DV 2016.
- Libraries: PyTorch, NumPy, pandas, tifffile.
- AI assistance: code scaffold, tests and proposal draft prepared with an AI assistant (Grok Bot); student must review, run and verify every result.

## IAM status (2026-11-08)
- Training role `<your-sagemaker-role>`: least privilege, train/val only, no test access.
- Studio role `<your-studio-role>`: **AdministratorAccess attached at user's explicit request**; test-data deny removed. Studio can technically read `data/test_holdout/`; the protocol still forbids it until freeze + approval. SCPs/resource policies may still restrict.
