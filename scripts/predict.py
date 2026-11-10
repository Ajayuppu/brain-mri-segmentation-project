"""Inference on arbitrary TIFF(s) -> binary 256x256 PNG mask(s) using a run dir (checkpoint, stats, val threshold)."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.data import read_image
from lggseg.model import UNet
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("images", nargs="+")
a = ap.parse_args(); meta = json.load(open(os.path.join(a.run, "run_meta.json"))); st = json.load(open(os.path.join(a.run, "channel_stats.json")))
m = UNet(base=meta["config"]["base_channels"]); m.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location="cpu")); m.eval()
os.makedirs(a.out_dir, exist_ok=True)
from PIL import Image
for p in a.images:
    x = (read_image(p) - np.array(st["mean"], np.float32)) / np.array(st["std"], np.float32)
    with torch.no_grad(): pr = torch.sigmoid(m(torch.from_numpy(x.transpose(2, 0, 1))[None]))[0, 0].numpy()
    Image.fromarray(((pr >= meta["selected_threshold"]) * 255).astype(np.uint8)).save(os.path.join(a.out_dir, os.path.basename(p).replace(".tif", "_pred.png")))
