"""Metrics per course protocol at original 256x256. Convention: both empty=1, exactly one empty=0."""
from __future__ import annotations
import numpy as np
import pandas as pd


def confusion(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, g).sum()); fp = int(np.logical_and(p, ~g).sum()); fn = int(np.logical_and(~p, g).sum())
    return tp, fp, fn


def dice(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 1.0 if d == 0 else 2 * tp / d


def iou(tp, fp, fn):
    d = tp + fp + fn
    return 1.0 if d == 0 else tp / d


def slice_records(preds, gts, patient_ids, slice_ids):
    rows = []
    for p, g, pid, sid in zip(preds, gts, patient_ids, slice_ids):
        assert p.shape == g.shape == (256, 256), (p.shape, g.shape)
        tp, fp, fn = confusion(p, g)
        rows.append(dict(patient_id=pid, slice_id=sid, tp=tp, fp=fp, fn=fn,
                         gt_pixels=int(g.sum()), pred_pixels=int(p.sum()),
                         gt_empty=int(g.sum() == 0), dice=dice(tp, fp, fn), iou=iou(tp, fp, fn)))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    TP, FP, FN = df.tp.sum(), df.fp.sum(), df.fn.sum()
    pos, emp = df[df.gt_empty == 0], df[df.gt_empty == 1]
    pat = df.groupby("patient_id").apply(lambda d: dice(d.tp.sum(), d.fp.sum(), d.fn.sum()), include_groups=False)
    return {
        "n_slices": int(len(df)),
        "pooled_foreground_dice": float(dice(TP, FP, FN)),
        "pooled_foreground_iou": float(iou(TP, FP, FN)),
        "mean_slice_dice": float(df.dice.mean()),
        "mean_slice_iou": float(df.iou.mean()),
        "positive_slice_dice": float(pos.dice.mean()) if len(pos) else float("nan"),
        "positive_slice_iou": float(pos.iou.mean()) if len(pos) else float("nan"),
        "n_positive": int(len(pos)), "n_empty": int(len(emp)),
        "empty_slices_with_fp": int((emp.fp > 0).sum()),
        "empty_fp_rate": float((emp.fp > 0).mean()) if len(emp) else float("nan"),
        "empty_fp_pixels_total": int(emp.fp.sum()),
        "empty_fp_pixels_mean": float(emp.fp.mean()) if len(emp) else float("nan"),
        "patient_dice_mean": float(pat.mean()), "patient_dice_median": float(pat.median()),
        "patient_dice_std": float(pat.std(ddof=1)) if len(pat) > 1 else 0.0,
        "patient_dice_min": float(pat.min()), "n_patients": int(len(pat)),
    }
