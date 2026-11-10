"""Freeze chosen runs BEFORE test evaluation: hashes checkpoints, normalization stats, run_meta, eval code."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.protocol import freeze
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", default="FROZEN.json")
a = ap.parse_args(); fr = freeze(a.runs, a.out); print("frozen", list(fr["runs"]), "->", a.out, "+ .sha256")
