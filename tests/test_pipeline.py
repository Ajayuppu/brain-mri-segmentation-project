import numpy as np, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lggseg.data import augment
from lggseg.model import UNet, n_params
from lggseg.losses import make_loss, soft_dice_loss

def test_aug_alignment():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = (np.random.rand(256, 256) > .9).astype(np.uint8); img = np.repeat(m[..., None], 3, 2).astype(np.float32) * np.array([1, 2, 3], np.float32)
        a, b = augment(img, m, rng)
        assert np.array_equal(a[..., 0], b) and np.array_equal(a[..., 2], 3 * b)  # aligned + channel order preserved

def test_unet_shape():
    m = UNet(base=8); y = m(torch.zeros(2, 3, 256, 256)); assert y.shape == (2, 1, 256, 256) and n_params(m) > 0

def test_losses_finite_and_posonly_ignores_empty():
    lg = torch.randn(4, 1, 32, 32); t = torch.zeros(4, 1, 32, 32); t[0, :, :5, :5] = 1
    for n in ["bce", "bce_dice", "bce_dice_pos", "weighted_bce"]:
        assert torch.isfinite(make_loss(n, 1.0, 10.0)(lg, t))
    assert soft_dice_loss(lg, torch.zeros_like(t), positive_only=True).item() == 0.0
    a = soft_dice_loss(lg, t, positive_only=True); b = soft_dice_loss(lg[:1], t[:1]); assert torch.allclose(a, b)
