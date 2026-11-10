import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.metrics import confusion, dice, iou, slice_records, summarize
Z = np.zeros((256, 256), np.uint8)

def one(r0, r1, c0, c1):
    m = Z.copy(); m[r0:r1, c0:c1] = 1; return m

def test_both_empty_is_one():
    assert dice(*confusion(Z, Z)) == 1.0 and iou(*confusion(Z, Z)) == 1.0

def test_exactly_one_empty_is_zero():
    m = one(0, 2, 0, 2)
    assert dice(*confusion(m, Z)) == 0.0 and dice(*confusion(Z, m)) == 0.0 and iou(*confusion(Z, m)) == 0.0

def test_partial_overlap():
    p, g = one(0, 10, 0, 10), one(0, 10, 5, 15)  # tp=50 fp=50 fn=50
    assert abs(dice(*confusion(p, g)) - 0.5) < 1e-12 and abs(iou(*confusion(p, g)) - 1/3) < 1e-12

def test_summary_separates_positive_and_empty():
    g = [Z, one(0, 10, 0, 10)]; p = [one(0, 1, 0, 3), Z]
    s = summarize(slice_records(p, g, ["a", "b"], ["1", "2"]))
    assert s["positive_slice_dice"] == 0.0 and s["empty_slices_with_fp"] == 1 and s["empty_fp_pixels_total"] == 3
    assert s["pooled_foreground_dice"] == 0.0 and s["mean_slice_dice"] == 0.0

def test_always_empty_reference():
    g = [Z, Z, one(0, 4, 0, 4)]
    s = summarize(slice_records([Z] * 3, g, ["a", "a", "b"], list("123")))
    assert abs(s["mean_slice_dice"] - 2/3) < 1e-12 and s["positive_slice_dice"] == 0.0 and s["empty_slices_with_fp"] == 0

def test_rejects_wrong_resolution():
    import pytest
    with pytest.raises(AssertionError):
        slice_records([np.zeros((128, 128))], [np.zeros((128, 128))], ["a"], ["1"])
