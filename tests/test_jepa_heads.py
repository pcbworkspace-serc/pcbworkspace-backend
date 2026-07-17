"""Fast smoke tests for jepa_heads.py — deliberately tiny (n_train in the
tens, not thousands) so this runs in seconds. The real validated numbers
(6.6deg / 1.1px / 95.8%) came from the full training run documented in
jepa_heads.py's __main__ block and jepa_heads_synthetic.pt's own metadata,
not from anything re-run here. This file exists to catch a broken shape or
a regressed loss, not to reproduce that training run in CI.
"""
import torch

import jepa_heads as jh


def test_feature_extractor_output_shape():
    from pcb_jepa_nn import load_model
    model, _ = load_model("best.pt", device="cpu")
    fx = jh.FrozenFeatureExtractor(model.backbone)
    x = torch.randn(2, 3, 64, 64)
    feats = fx(x)
    assert feats.shape == (2, jh.FEATURE_DIM)


def test_align_cnn_and_head_shapes():
    cnn = jh.AlignCNN()
    head = jh.AlignHead()
    x = torch.randn(3, 3, jh.WIN_SIZE, jh.WIN_SIZE)
    out = head(cnn(x))
    assert out.shape == (3, 4)  # sin, cos, dx_norm, dy_norm
    # sin/cos columns should sit on the unit circle
    norms = out[:, :2].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-4)


def test_validate_head_shape():
    head = jh.ValidateHead()
    feats = torch.randn(5, jh.FEATURE_DIM)
    logits = head(feats)
    assert logits.shape == (5, 2)


def test_synthetic_generators_are_self_consistent():
    img, theta, dx, dy = jh.gen_alignment_sample()
    assert -180 <= theta <= 180
    assert -jh.MAX_OFFSET_PX <= dx <= jh.MAX_OFFSET_PX
    assert -jh.MAX_OFFSET_PX <= dy <= jh.MAX_OFFSET_PX
    assert img.size == (jh.WIN_SIZE, jh.WIN_SIZE)

    img2, label, reason = jh.gen_validation_sample()
    assert label in ("pass", "fail")
    assert img2.size == (jh.WIN_SIZE, jh.WIN_SIZE)


def test_align_training_loop_reduces_loss():
    """Tiny run (not the real training config) - just proves the training
    loop is wired correctly and loss trends down, not a quality bar."""
    torch.manual_seed(0)
    import random
    random.seed(0)

    cnn = jh.AlignCNN()
    head = jh.AlignHead()
    opt = torch.optim.Adam(list(cnn.parameters()) + list(head.parameters()), lr=2e-3)

    def batch(n):
        imgs, thetas, dxs, dys = [], [], [], []
        for _ in range(n):
            img, theta, dx, dy = jh.gen_alignment_sample()
            imgs.append(jh._to_tensor(img)); thetas.append(theta)
            dxs.append(dx / jh.MAX_OFFSET_PX); dys.append(dy / jh.MAX_OFFSET_PX)
        return (torch.stack(imgs), torch.tensor(thetas, dtype=torch.float32),
                torch.tensor(dxs, dtype=torch.float32), torch.tensor(dys, dtype=torch.float32))

    x, theta_t, dx_t, dy_t = batch(64)
    import torch.nn.functional as F

    def step_loss():
        out = head(cnn(x))
        loss_ang = F.mse_loss(out[:, 0], torch.sin(torch.deg2rad(theta_t))) + \
                   F.mse_loss(out[:, 1], torch.cos(torch.deg2rad(theta_t)))
        loss_off = F.smooth_l1_loss(out[:, 2], dx_t) + F.smooth_l1_loss(out[:, 3], dy_t)
        return loss_ang + loss_off

    first_loss = step_loss().item()
    for _ in range(30):
        opt.zero_grad()
        loss = step_loss()
        loss.backward()
        opt.step()
    last_loss = step_loss().item()

    assert last_loss < first_loss  # overfitting a tiny fixed batch should trivially reduce loss


def test_save_and_load_roundtrip(tmp_path):
    cnn = jh.AlignCNN().eval()  # dropout must be off for a deterministic before/after comparison
    head = jh.AlignHead().eval()
    validate_head = jh.ValidateHead().eval()
    path = str(tmp_path / "roundtrip.pt")

    jh.save_checkpoint(cnn, head, validate_head,
                        {"val_angle_mae_deg": 1.0}, {"val_accuracy": 1.0}, path=path)
    loaded_cnn, loaded_head, loaded_validate, meta = jh.load_heads(path=path)

    assert loaded_cnn is not None and loaded_head is not None and loaded_validate is not None
    assert meta["align_metrics"]["val_angle_mae_deg"] == 1.0

    x = torch.randn(1, 3, jh.WIN_SIZE, jh.WIN_SIZE)
    with torch.no_grad():
        original_out = head(cnn(x))
        loaded_out = loaded_head(loaded_cnn(x))
    assert torch.allclose(original_out, loaded_out)


def test_load_heads_missing_file_returns_none():
    cnn, head, validate_head, meta = jh.load_heads(path="does_not_exist_anywhere.pt")
    assert cnn is None and head is None and validate_head is None and meta is None
