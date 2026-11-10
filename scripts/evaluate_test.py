"""ONE-TIME test evaluation of frozen runs + always-empty reference (see src/lggseg/protocol.py for guarantees/limits)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.protocol import run_once


def evaluate(fr, mark_labels_loaded, data_root, out_dir):
    import torch
    from lggseg import data as D
    from lggseg.data import load_manifest
    from lggseg.model import UNet
    from lggseg.metrics import slice_records, summarize
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_manifest(data_root, "test"); probs_all = {}
    for run, spec in fr["runs"].items():  # predictions first (images only, normalization = frozen train stats)
        m = UNet(base=spec["config"]["base_channels"]).to(dev)
        m.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=dev)); m.eval()
        mean = np.asarray(spec["normalization"]["mean"], np.float32); std = np.asarray(spec["normalization"]["std"], np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(df), 16):  # PASS 1: images only; mask_path is never opened here
                xb = np.stack([((D.read_image(os.path.join(data_root, p)) - mean) / std).transpose(2, 0, 1) for p in df.image_path[i:i + 16]])
                out.append(torch.sigmoid(m(torch.from_numpy(xb.astype(np.float32)).to(dev))).cpu().numpy()[:, 0])
        probs_all[run] = np.concatenate(out)
    mark_labels_loaded()  # ledger records label access BEFORE the first mask read (PASS 2)
    gts = np.stack([D.read_mask(os.path.join(data_root, p)) for p in df.mask_path])
    res = {"always_empty": summarize(slice_records(np.zeros_like(gts), gts, df.patient_id, df.slice_id))}
    for run, spec in fr["runs"].items():
        sr = slice_records(probs_all[run] >= spec["threshold"], gts, df.patient_id, df.slice_id)
        sr.to_csv(os.path.join(out_dir, os.path.basename(run) + "_slices.csv"), index=False); res[run] = summarize(sr)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); ap.add_argument("--frozen", default="FROZEN.json")
    ap.add_argument("--out-dir", default="evidence/test"); ap.add_argument("--i-confirm-method-is-frozen", action="store_true")
    ap.add_argument("--retry-reason", help="only after a logged failure that occurred before test labels were loaded")
    a = ap.parse_args()
    if not a.i_confirm_method_is_frozen: raise SystemExit("Refusing: pass --i-confirm-method-is-frozen")
    res = run_once(a.out_dir, a.frozen, lambda fr, mk: evaluate(fr, mk, a.data_root, a.out_dir), a.retry_reason)
    print(json.dumps(res, indent=2))
