"""Metadata-only audit. Uses train/val label statistics; for TEST uses only ids, counts and hashes (no label stats)."""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); ap.add_argument("--out", default="evidence/metadata_audit.json")
a = ap.parse_args()
r = lambda f: pd.read_csv(os.path.join(a.data_root, f))
full, ps = r("manifest.csv"), r("patient_splits.csv")
parts = {s: r(f"{s}_manifest.csv") for s in ("train", "val", "test")}
out = {"expected": {"train": [77, 2763], "val": [17, 598], "test": [16, 568]}, "checks": {}, "train_val_stats": {}}
for s, d in parts.items():
    out["checks"][s] = {"patients": int(d.patient_id.nunique()), "pairs": int(len(d)),
                        "all_rows_split": bool((d.split == s).all()),
                        "dims_all_256x256x3": bool(((d.height == 256) & (d.width == 256) & (d.channels == 3)).all()),
                        "dup_image_hash": int(d.image_sha256.duplicated().sum())}
sets = {s: set(d.patient_id) for s, d in parts.items()}
out["checks"]["patient_overlap"] = {f"{x}-{y}": len(sets[x] & sets[y]) for x, y in [("train", "val"), ("train", "test"), ("val", "test")]}
out["checks"]["union_equals_full_manifest"] = int(sum(len(d) for d in parts.values())) == len(full)
allh = pd.concat(parts.values()).image_sha256
out["checks"]["cross_split_duplicate_image_hashes"] = int(allh.duplicated().sum())
pc = pd.concat(parts.values()).groupby("patient_id").size()
out["checks"]["patient_splits_pair_counts_match"] = bool((ps.set_index("patient_id").pair_count == pc.reindex(ps.patient_id).values).all())
for s in ("train", "val"):
    d = parts[s]; pos = d[d.has_foreground == 1]
    per_pat = d.groupby("patient_id").has_foreground.mean()
    out["train_val_stats"][s] = {
        "positive_slices": int(len(pos)), "empty_slices": int((d.has_foreground == 0).sum()),
        "positive_fraction": round(float(d.has_foreground.mean()), 4),
        "fg_pixels_quantiles_positive": {str(q): float(pos.foreground_pixels.quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        "fg_fraction_of_all_pixels": float(d.foreground_pixels.sum() / (len(d) * 65536)),
        "small_lesions_lt_100px": int((pos.foreground_pixels < 100).sum()),
        "patients_with_no_positive": int((per_pat == 0).sum()),
        "per_patient_positive_fraction_min_med_max": [float(per_pat.min()), float(per_pat.median()), float(per_pat.max())]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))
