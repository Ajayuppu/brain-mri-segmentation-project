"""Freeze + one-time test-evaluation protocol.

What this DOES guarantee: any change to frozen checkpoints, normalization statistics, run metadata,
configs, the freeze record, or the hashed source files is detected and evaluation refuses; a completed
or started test evaluation cannot be started again through this code path.
What it does NOT guarantee: it cannot stop someone editing this code, deleting the ledger, or computing
test metrics with another script. Integrity ultimately rests on the honesty of the experimenter and on
the append-only ledger + Google Doc log being disclosed in the report.
"""
from __future__ import annotations
import glob, hashlib, json, os, socket, time, traceback

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_FILES = ["src/lggseg/data.py", "src/lggseg/model.py", "src/lggseg/metrics.py", "src/lggseg/protocol.py",
              "scripts/evaluate_test.py"]
RUN_ARTIFACTS = ["best.pt", "channel_stats.json", "run_meta.json"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class ProtocolError(SystemExit):
    pass


def freeze(runs, out, pkg_root=PKG):
    if os.path.exists(out):
        raise ProtocolError(f"{out} exists; freezing is one-shot.")
    fr = {"frozen_at": now(), "runs": {}, "code_sha256": {}}
    for c in CODE_FILES:
        p = os.path.join(pkg_root, c)
        if os.path.exists(p): fr["code_sha256"][c] = sha256_file(p)
    for r in runs:
        meta = json.load(open(os.path.join(r, "run_meta.json")))
        if meta.get("smoke_test"): raise ProtocolError(f"{r} is a smoke test; cannot freeze")
        art = {a: sha256_file(os.path.join(r, a)) for a in RUN_ARTIFACTS}
        if meta.get("checkpoint_sha256") and meta["checkpoint_sha256"] != art["best.pt"]:
            raise ProtocolError(f"{r}: checkpoint differs from hash recorded at training time")
        stats = json.load(open(os.path.join(r, "channel_stats.json")))
        if stats.get("fitted_on") != "train": raise ProtocolError(f"{r}: normalization stats not fitted on train")
        fr["runs"][os.path.abspath(r)] = {"artifact_sha256": art, "threshold": meta["selected_threshold"],
                                          "config": meta["config"], "normalization": stats}
    tmp = out + ".tmp"
    json.dump(fr, open(tmp, "w"), indent=2); os.replace(tmp, out)
    with open(out + ".sha256", "w") as f: f.write(sha256_file(out) + "\n")
    return fr


def verify_frozen(frozen_path, pkg_root=PKG):
    """Return frozen dict if every recorded hash matches; else raise listing all mismatches."""
    sp = frozen_path + ".sha256"
    if not os.path.exists(frozen_path) or not os.path.exists(sp): raise ProtocolError("frozen record or its .sha256 missing")
    errs = []
    if open(sp).read().strip() != sha256_file(frozen_path): errs.append("FROZEN.json modified after freeze")
    fr = json.load(open(frozen_path))
    for c, h in fr["code_sha256"].items():
        p = os.path.join(pkg_root, c)
        if not os.path.exists(p) or sha256_file(p) != h: errs.append(f"code changed since freeze: {c}")
    for r, spec in fr["runs"].items():
        for a, h in spec["artifact_sha256"].items():
            p = os.path.join(r, a)
            if not os.path.exists(p) or sha256_file(p) != h: errs.append(f"artifact changed: {r}/{a}")
        st = os.path.join(r, "channel_stats.json")
        if os.path.exists(st) and json.load(open(st)) != spec["normalization"]: errs.append(f"normalization differs: {r}")
    if errs: raise ProtocolError("Refusing test evaluation:\n  " + "\n  ".join(errs))
    return fr


def ledger_path(out_dir):
    return os.path.join(out_dir, "TEST_EVAL_LEDGER.json")


def _write_ledger(path, led):
    tmp = path + ".tmp"; json.dump(led, open(tmp, "w"), indent=2); os.replace(tmp, path)


def acquire_once(out_dir, frozen_path, retry_reason=None):
    """Atomically claim the single test evaluation. Lock created with O_EXCL BEFORE any test data is read.
    A completed evaluation can never be re-run. A failed attempt may be retried only with a written
    reason, and only if it failed before any test labels were loaded; every attempt is kept in the ledger."""
    os.makedirs(out_dir, exist_ok=True)
    lock, lp = os.path.join(out_dir, "TEST_EVAL.lock"), ledger_path(out_dir)
    fsha = sha256_file(frozen_path)
    if os.path.exists(lp):
        led = json.load(open(lp)); last = led["attempts"][-1]
        if last["status"] == "completed": raise ProtocolError("Test evaluation already completed; refusing to re-run.")
        if last["status"] == "started": raise ProtocolError("A test evaluation is marked in progress (or crashed hard). Inspect ledger; do not re-run silently.")
        if last["status"] == "failed":
            if not retry_reason: raise ProtocolError("Previous attempt failed; re-run requires --retry-reason and is logged.")
            if last.get("test_labels_loaded"): raise ProtocolError("Previous failed attempt had already loaded test labels; retry refused. Disclose in report.")
            if led["frozen_sha256"] != fsha: raise ProtocolError("Frozen record differs from the one used in the failed attempt.")
        os.remove(lock) if os.path.exists(lock) else None
    else:
        if retry_reason: raise ProtocolError("--retry-reason given but there is no failed attempt.")
        led = {"frozen_sha256": fsha, "attempts": []}
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ProtocolError("TEST_EVAL.lock exists; another or previous evaluation claimed the test set.")
    os.write(fd, f"{now()} pid={os.getpid()} host={socket.gethostname()}\n".encode()); os.close(fd)
    led["attempts"].append({"started_at": now(), "status": "started", "test_labels_loaded": False, "retry_reason": retry_reason})
    _write_ledger(lp, led)
    return led


def mark(out_dir, **kw):
    lp = ledger_path(out_dir); led = json.load(open(lp)); led["attempts"][-1].update(kw); _write_ledger(lp, led)


def run_once(out_dir, frozen_path, body, retry_reason=None, pkg_root=PKG):
    """verify -> claim -> body(fr, mark_labels_loaded) -> record outcome. Exceptions are recorded as failed."""
    fr = verify_frozen(frozen_path, pkg_root)
    acquire_once(out_dir, frozen_path, retry_reason)
    try:
        res = body(fr, lambda: mark(out_dir, test_labels_loaded=True))
        rp = os.path.join(out_dir, "test_results.json")
        json.dump(res, open(rp + ".tmp", "w"), indent=2); os.replace(rp + ".tmp", rp)
        mark(out_dir, status="completed", finished_at=now(), results_sha256=sha256_file(rp))
        return res
    except BaseException as e:
        mark(out_dir, status="failed", finished_at=now(), error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
        raise
