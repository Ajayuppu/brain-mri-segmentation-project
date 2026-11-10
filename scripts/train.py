"""Train one condition; select checkpoint and threshold on VALIDATION only. Never touches test."""
import argparse, json, os, time, platform, random, hashlib
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.data import load_manifest, fit_channel_stats, save_stats, SegDataset
from lggseg.model import UNet, n_params
from lggseg.losses import make_loss
from lggseg.metrics import slice_records, summarize


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict_probs(model, loader, dev):
    model.eval(); out = []
    for x, _, _ in loader:
        out.append(torch.sigmoid(model(x.to(dev))).cpu().numpy()[:, 0])
    return np.concatenate(out)


def val_score(probs, gts, df, thr):
    s = summarize(slice_records(probs >= thr, gts, df.patient_id, df.slice_id))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="runs"); ap.add_argument("--seed", type=int); ap.add_argument("--max-train", type=int, default=0,
                    help="smoke-test only; 0 = full training partition")
    ap.add_argument("--override", nargs="*", default=[], help="key=value (YAML-parsed), recorded in run_meta")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.seed is not None: cfg["seed"] = a.seed
    for kv in a.override:
        k, v = kv.split("=", 1); cfg[k] = yaml.safe_load(v)
    if a.max_train == 0 and a.override: cfg["name"] += "_ovr"
    seed_all(cfg["seed"])
    run = os.path.join(a.out_root, f"{cfg['name']}_s{cfg['seed']}"); os.makedirs(run, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = load_manifest(a.data_root, "train"), load_manifest(a.data_root, "val")
    if a.max_train: tr = tr.iloc[:a.max_train]; va = va.iloc[:max(8, a.max_train // 2)]
    stats_path = os.path.join(a.out_root, "train_channel_stats.json")
    if a.max_train or not os.path.exists(stats_path):
        stats = fit_channel_stats(a.data_root, tr)
        if not a.max_train: save_stats(stats, stats_path)
    else:
        stats = json.load(open(stats_path))
    save_stats(stats, os.path.join(run, "channel_stats.json"))
    pos_weight = cfg.get("pos_weight")
    if cfg["loss"] == "weighted_bce" and pos_weight in (None, "auto"):
        fg = tr.foreground_pixels.sum() / (len(tr) * 65536); pos_weight = float(min((1 - fg) / fg, cfg.get("pos_weight_cap", 1e9)))
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    tl = DataLoader(SegDataset(a.data_root, tr, stats, train=True, seed=cfg["seed"]), batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg.get("workers", 4), generator=g, drop_last=True)
    vl = DataLoader(SegDataset(a.data_root, va, stats), batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 4))
    from lggseg.data import read_mask
    vgts = np.stack([read_mask(os.path.join(a.data_root, p)) for p in va.mask_path])
    model = UNet(base=cfg["base_channels"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    loss_fn = make_loss(cfg["loss"], cfg.get("dice_weight", 1.0), pos_weight)
    if len(tl) == 0: raise SystemExit("no training batches (batch_size > n_train with drop_last)")
    sel = cfg["selection_metric"]; best = -1; log = []; t0 = time.time(); bad = 0
    for ep in range(cfg["epochs"]):
        model.train(); tot = 0
        for x, y, _ in tl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); l = loss_fn(model(x), y); l.backward(); opt.step(); tot += l.item() * len(x)
        sched.step()
        probs = predict_probs(model, vl, dev); s = val_score(probs, vgts, va, 0.5)
        rec = {"epoch": ep, "train_loss": tot / len(tl.dataset), "val@0.5": s, "elapsed_s": time.time() - t0}; log.append(rec)
        print(json.dumps({"epoch": ep, "loss": rec["train_loss"], sel: s[sel], "empty_fp": s["empty_slices_with_fp"]}), flush=True)
        if s[sel] > best:
            best = s[sel]; bad = 0; torch.save(model.state_dict(), os.path.join(run, "best.pt"))
        else:
            bad += 1
            if bad >= cfg.get("patience", 10**9): break
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=dev))
    probs = predict_probs(model, vl, dev)
    grid = {f"{t:.2f}": val_score(probs, vgts, va, t) for t in cfg["threshold_grid"]}
    thr = max(grid, key=lambda k: grid[k][sel])
    meta = {"config": cfg, "pos_weight_used": pos_weight, "seed": cfg["seed"], "selected_threshold": float(thr),
            "selection_metric": sel, "val_at_selected": grid[thr], "val_threshold_grid": grid, "params": n_params(model),
            "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "torch": torch.__version__,
            "python": platform.python_version(), "train_time_s": time.time() - t0, "epochs_run": len(log),
            "smoke_test": bool(a.max_train), "n_train": len(tr), "n_val": len(va),
            "checkpoint_sha256": hashlib.sha256(open(os.path.join(run, "best.pt"), "rb").read()).hexdigest()}
    json.dump(log, open(os.path.join(run, "train_log.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(run, "run_meta.json"), "w"), indent=2)
    print("VAL (selected thr)", thr, json.dumps(grid[thr]))


if __name__ == "__main__":
    main()
