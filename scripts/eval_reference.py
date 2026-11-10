"""Always-empty trivial reference on VAL (test only via evaluate_test.py after freeze)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.data import load_manifest, read_mask
from lggseg.metrics import slice_records, summarize
ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); ap.add_argument("--out", default="evidence/always_empty_val.json")
a = ap.parse_args()
df = load_manifest(a.data_root, "val")
g = [read_mask(os.path.join(a.data_root, p)) for p in df.mask_path]
s = summarize(slice_records([np.zeros_like(x) for x in g], g, df.patient_id, df.slice_id))
json.dump(s, open(a.out, "w"), indent=2); print(json.dumps(s, indent=2))
