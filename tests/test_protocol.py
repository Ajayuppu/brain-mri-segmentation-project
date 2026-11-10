"""Synthetic-only tests (no real test images/masks are read)."""
import json, os, sys, shutil, subprocess, pytest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg import protocol as P

PKG = P.PKG


def make_run(d, smoke=False):
    os.makedirs(d)
    import torch
    from lggseg.model import UNet
    torch.save(UNet(base=4).state_dict(), os.path.join(d, "best.pt"))
    json.dump({"mean": [1, 2, 3], "std": [1, 1, 1], "fitted_on": "train", "n_images": 4}, open(os.path.join(d, "channel_stats.json"), "w"))
    json.dump({"smoke_test": smoke, "selected_threshold": 0.5, "config": {"base_channels": 4},
               "checkpoint_sha256": P.sha256_file(os.path.join(d, "best.pt"))}, open(os.path.join(d, "run_meta.json"), "w"))
    return d


@pytest.fixture
def setup(tmp_path):
    run = make_run(str(tmp_path / "run"))
    fz = str(tmp_path / "FROZEN.json"); P.freeze([run], fz)
    return run, fz, str(tmp_path / "out")


def ok_body(fr, mk):
    mk(); return {"x": 1}


def test_freeze_records_normalization_hash(setup):
    run, fz, _ = setup
    spec = json.load(open(fz))["runs"][os.path.abspath(run)]
    assert spec["artifact_sha256"]["channel_stats.json"] == P.sha256_file(os.path.join(run, "channel_stats.json"))
    assert spec["normalization"]["mean"] == [1, 2, 3]
    P.verify_frozen(fz)


def test_tampered_normalization_refused(setup):
    run, fz, out = setup
    p = os.path.join(run, "channel_stats.json"); s = json.load(open(p)); s["mean"][0] = 1.0001; json.dump(s, open(p, "w"))
    with pytest.raises(P.ProtocolError, match="channel_stats"):
        P.run_once(out, fz, ok_body)
    assert not os.path.exists(out)  # refused before claiming the test set


def test_tampered_checkpoint_meta_and_freeze_record_refused(setup, tmp_path):
    run, fz, out = setup
    open(os.path.join(run, "best.pt"), "ab").write(b"x")
    with pytest.raises(P.ProtocolError, match="best.pt"): P.verify_frozen(fz)
    r2 = make_run(str(tmp_path / "r2")); f2 = str(tmp_path / "F2.json"); P.freeze([r2], f2)
    d = json.load(open(f2)); d["runs"][os.path.abspath(r2)]["threshold"] = 0.1; json.dump(d, open(f2, "w"))
    with pytest.raises(P.ProtocolError, match="modified after freeze"): P.verify_frozen(f2)


def test_code_change_detected(tmp_path):
    pkg = tmp_path / "pkg"; shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("runs", "evidence", "*.zip", "__pycache__", ".pytest_cache"))
    run = make_run(str(tmp_path / "run")); fz = str(tmp_path / "F.json"); P.freeze([run], fz, pkg_root=str(pkg))
    with open(pkg / "src/lggseg/metrics.py", "a") as f: f.write("\n# edit\n")
    with pytest.raises(P.ProtocolError, match="metrics.py"): P.verify_frozen(fz, pkg_root=str(pkg))


def test_smoke_and_nontrain_stats_cannot_freeze(tmp_path):
    with pytest.raises(P.ProtocolError, match="smoke"): P.freeze([make_run(str(tmp_path / "s"), smoke=True)], str(tmp_path / "a.json"))
    r = make_run(str(tmp_path / "v")); p = os.path.join(r, "channel_stats.json"); s = json.load(open(p)); s["fitted_on"] = "val"; json.dump(s, open(p, "w"))
    with pytest.raises(P.ProtocolError, match="not fitted on train"): P.freeze([r], str(tmp_path / "b.json"))


def test_freeze_is_one_shot(setup):
    run, fz, _ = setup
    with pytest.raises(P.ProtocolError, match="one-shot"): P.freeze([run], fz)


def test_second_evaluation_refused(setup):
    run, fz, out = setup
    assert P.run_once(out, fz, ok_body) == {"x": 1}
    led = json.load(open(P.ledger_path(out))); assert led["attempts"][-1]["status"] == "completed"
    with pytest.raises(P.ProtocolError, match="already completed"): P.run_once(out, fz, ok_body)
    os.remove(os.path.join(out, "TEST_EVAL.lock"))  # deleting the lock alone does not reopen evaluation
    with pytest.raises(P.ProtocolError, match="already completed"): P.run_once(out, fz, ok_body)


def test_lock_without_ledger_refused(setup):
    run, fz, out = setup
    os.makedirs(out); open(os.path.join(out, "TEST_EVAL.lock"), "w").write("x")
    with pytest.raises(P.ProtocolError, match="lock exists"): P.run_once(out, fz, ok_body)


def test_failure_before_labels_retry_only_with_reason(setup):
    run, fz, out = setup
    def boom(fr, mk): raise RuntimeError("CUDA OOM")
    with pytest.raises(RuntimeError): P.run_once(out, fz, boom)
    a = json.load(open(P.ledger_path(out)))["attempts"][-1]; assert a["status"] == "failed" and not a["test_labels_loaded"]
    with pytest.raises(P.ProtocolError, match="retry-reason"): P.run_once(out, fz, ok_body)
    assert P.run_once(out, fz, ok_body, retry_reason="OOM before labels; batch reduced") == {"x": 1}
    led = json.load(open(P.ledger_path(out))); assert [x["status"] for x in led["attempts"]] == ["failed", "completed"]


def test_failure_after_labels_never_retried(setup):
    run, fz, out = setup
    def late(fr, mk): mk(); raise RuntimeError("crash after labels")
    with pytest.raises(RuntimeError): P.run_once(out, fz, late)
    with pytest.raises(P.ProtocolError, match="already loaded test labels"): P.run_once(out, fz, ok_body, retry_reason="x")


def test_hard_crash_state_started_refused(setup):
    run, fz, out = setup
    P.verify_frozen(fz); P.acquire_once(out, fz)  # simulate kill -9: ledger stays 'started'
    with pytest.raises(P.ProtocolError, match="in progress"): P.run_once(out, fz, ok_body, retry_reason="x")


def test_cli_requires_confirmation_and_freeze(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(PKG, "scripts/evaluate_test.py"), "--data-root", "x", "--frozen", str(tmp_path / "none.json"),
                        "--out-dir", str(tmp_path / "o"), "--i-confirm-method-is-frozen"], capture_output=True, text=True)
    assert r.returncode != 0 and "missing" in (r.stderr + r.stdout) and not (tmp_path / "o").exists()


def test_end_to_end_synthetic_eval(tmp_path):
    import tifffile, pandas as pd, importlib.util
    root = tmp_path / "data"; rows = []
    for i, has in enumerate([0, 1]):
        d = root / "test/SYN_P"; (d / "images").mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(exist_ok=True)
        img = (np.random.default_rng(i).random((256, 256, 3)) * 255).astype(np.uint8); m = np.zeros((256, 256), np.uint8); m[:10, :10] = 255 * has
        tifffile.imwrite(d / f"images/s{i}.tif", img); tifffile.imwrite(d / f"masks/s{i}_mask.tif", m)
        rows.append(dict(patient_id="SYN_P", slice_id=str(i), split="test", image_path=f"test/SYN_P/images/s{i}.tif", mask_path=f"test/SYN_P/masks/s{i}_mask.tif"))
    pd.DataFrame(rows).to_csv(root / "test_manifest.csv", index=False)
    run = make_run(str(tmp_path / "run")); fz = str(tmp_path / "F.json"); P.freeze([run], fz); out = str(tmp_path / "out")
    spec = importlib.util.spec_from_file_location("ev", os.path.join(PKG, "scripts/evaluate_test.py")); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    res = P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out))
    assert res["always_empty"]["n_slices"] == 2 and res["always_empty"]["positive_slice_dice"] == 0.0
    a = json.load(open(P.ledger_path(out)))["attempts"][-1]; assert a["status"] == "completed" and a["test_labels_loaded"]
    with pytest.raises(P.ProtocolError): P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out))


def _syn_root(tmp_path, n=3):
    import tifffile, pandas as pd
    root = tmp_path / "data"; rows = []
    d = root / "test/SYN_P"; (d / "images").mkdir(parents=True); (d / "masks").mkdir()
    for i in range(n):
        tifffile.imwrite(d / f"images/s{i}.tif", (np.random.default_rng(i).random((256, 256, 3)) * 255).astype(np.uint8))
        m = np.zeros((256, 256), np.uint8); m[:8, :8] = 255 * (i % 2); tifffile.imwrite(d / f"masks/s{i}_mask.tif", m)
        rows.append(dict(patient_id="SYN_P", slice_id=str(i), split="test", image_path=f"test/SYN_P/images/s{i}.tif", mask_path=f"test/SYN_P/masks/s{i}_mask.tif"))
    pd.DataFrame(rows).to_csv(root / "test_manifest.csv", index=False)
    return root


def _load_ev():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", os.path.join(PKG, "scripts/evaluate_test.py")); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev); return ev


def test_real_eval_path_reads_no_mask_before_ledger_mark(tmp_path, monkeypatch):
    from lggseg import data as D
    root = _syn_root(tmp_path); run = make_run(str(tmp_path / "run")); fz = str(tmp_path / "F.json"); P.freeze([run], fz); out = str(tmp_path / "out")
    events = []; ri, rm = D.read_image, D.read_mask
    monkeypatch.setattr(D, "read_image", lambda p: (events.append(("image", p)), ri(p))[1])
    def spy_mask(p):
        led = json.load(open(P.ledger_path(out)))["attempts"][-1]
        events.append(("mask", p, led["test_labels_loaded"])); return rm(p)
    monkeypatch.setattr(D, "read_mask", spy_mask)
    real_open = open
    import builtins
    def guard_open(f, *a, **k):  # any mask file opened by any route before the ledger mark fails the test
        if isinstance(f, (str, os.PathLike)) and "_mask.tif" in str(f) and os.path.exists(P.ledger_path(out)):
            assert json.load(real_open(P.ledger_path(out)))["attempts"][-1]["test_labels_loaded"], "mask opened before ledger mark"
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", guard_open)
    ev = _load_ev(); P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out))
    kinds = [e[0] for e in events]
    assert kinds.count("image") == 3 and kinds.count("mask") == 3
    assert max(i for i, k in enumerate(kinds) if k == "image") < min(i for i, k in enumerate(kinds) if k == "mask")
    assert all(e[2] for e in events if e[0] == "mask")  # ledger already said labels loaded at every mask read


def test_failure_during_real_mask_read_blocks_retry(tmp_path, monkeypatch):
    from lggseg import data as D
    root = _syn_root(tmp_path); run = make_run(str(tmp_path / "run")); fz = str(tmp_path / "F.json"); P.freeze([run], fz); out = str(tmp_path / "out")
    def boom(p): raise OSError("disk error reading mask")
    monkeypatch.setattr(D, "read_mask", boom)
    ev = _load_ev()
    with pytest.raises(OSError): P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out))
    a = json.load(open(P.ledger_path(out)))["attempts"][-1]; assert a["status"] == "failed" and a["test_labels_loaded"]
    with pytest.raises(P.ProtocolError, match="already loaded test labels"):
        P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out), retry_reason="try again")


def test_failure_during_image_pass_allows_logged_retry(tmp_path, monkeypatch):
    from lggseg import data as D
    root = _syn_root(tmp_path); run = make_run(str(tmp_path / "run")); fz = str(tmp_path / "F.json"); P.freeze([run], fz); out = str(tmp_path / "out")
    ri = D.read_image; monkeypatch.setattr(D, "read_image", lambda p: (_ for _ in ()).throw(OSError("image read failed")))
    ev = _load_ev()
    with pytest.raises(OSError): P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out))
    assert not json.load(open(P.ledger_path(out)))["attempts"][-1]["test_labels_loaded"]
    monkeypatch.setattr(D, "read_image", ri)
    res = P.run_once(out, fz, lambda fr, mk: ev.evaluate(fr, mk, str(root), out), retry_reason="image IO error before labels")
    assert res["always_empty"]["n_slices"] == 3
