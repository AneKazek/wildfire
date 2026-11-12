"""Segmentation losses that play well with ~0.2% positives per patch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def _dice(p, t, dims=(1, 2, 3)) -> torch.Tensor:
    inter = (p * t).sum(dims)
    den = p.sum(dims) + t.sum(dims)
    return (2 * inter + EPS) / (den + EPS)


def bce_dice(logits, y, valid, pos_weight=5.0, dice_w=1.0):
    bce = F.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    d = _dice(torch.sigmoid(logits) * valid, y * valid)
    return bce + dice_w * (1 - d.mean())


def tversky(logits, y, valid, alpha=0.3, beta=0.7, gamma=1.0):
    p = torch.sigmoid(logits) * valid
    t = y * valid
    dims = (1, 2, 3)
    tp = (p * t).sum(dims)
    fp = (p * (1 - t)).sum(dims)
    fn = ((1 - p) * t).sum(dims)
    ti = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
    return (1 - ti).pow(gamma).mean()


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jac = 1.0 - inter / union
    if p > 1:
        jac[1:p] = jac[1:p] - jac[0:p - 1]
    return jac


def lovasz_hinge(logits, y, valid):
    losses = []
    for lg, lb, vb in zip(logits, y, valid):
        m = vb.reshape(-1) > 0.5
        if m.sum() == 0:
            continue
        lg, lb = lg.reshape(-1)[m], lb.reshape(-1)[m]
        signs = 2.0 * lb - 1.0
        errors = 1.0 - lg * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = lb[perm]
        losses.append(torch.dot(F.relu(errors_sorted), _lovasz_grad(gt_sorted)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def moment(logits, y, valid, scale: float = 64.0, floor: float = 0.05, cap: float = 4.0):
    """Match the first two moments of the predicted fire mass to the target.

    The metric is per-patch Dice on a ~6 px blob, so what decides it is where the
    mass sits and how far it spreads. Segmentation losses are blind to both: a
    diffuse-but-centred blob has the same Dice as a compact one of the same
    overlap, and nothing pulls the mass into the blob. This term does.

    `floor` removes the background probability carpet (BCE with pos_weight on
    0.2% positives leaves p~0.01 everywhere, whose total mass rivals the fire);
    `cap` keeps the early "all pixels at 0.5" phase from dominating the loss.
    Units: pixels squared, divided by `scale` (8 px mean+spread error ~1.0).
    """
    p = torch.relu(torch.sigmoid(logits) - floor) * valid
    t = y * valid
    h, w = logits.shape[-2:]
    ys = torch.arange(h, device=logits.device, dtype=torch.float32) - (h - 1) / 2
    xs = torch.arange(w, device=logits.device, dtype=torch.float32) - (w - 1) / 2

    def stats(a):
        s = a.sum(dim=(1, 2, 3), keepdim=True).clamp_min(EPS)
        my = (a * ys.view(1, 1, h, 1)).sum(dim=(1, 2, 3), keepdim=True) / s
        mx = (a * xs.view(1, 1, 1, w)).sum(dim=(1, 2, 3), keepdim=True) / s
        vy = (a * (ys.view(1, 1, h, 1) - my) ** 2).sum(dim=(1, 2, 3), keepdim=True) / s
        vx = (a * (xs.view(1, 1, 1, w) - mx) ** 2).sum(dim=(1, 2, 3), keepdim=True) / s
        return my, mx, torch.sqrt(vy + 0.25), torch.sqrt(vx + 0.25)

    d = sum(((a - b) ** 2).mean() for a, b in zip(stats(p), stats(t))) / scale
    return d.clamp(max=cap).mean()


def make_loss(lcfg: dict):
    name = lcfg.get("name", "bce_dice")
    if name == "bce_dice":
        pw, dw = lcfg.get("pos_weight", 5.0), lcfg.get("dice_weight", 1.0)
        return lambda lg, y, v: bce_dice(lg, y, v, pw, dw)
    if name == "tversky" or name == "focal_tversky":
        a, b, g = lcfg.get("alpha", 0.3), lcfg.get("beta", 0.7), lcfg.get("gamma", 1.0)
        g = g if name == "focal_tversky" else 1.0
        return lambda lg, y, v: tversky(lg, y, v, a, b, g)
    if name == "lovasz":
        return lovasz_hinge
    if name == "dice":
        return lambda lg, y, v: 1 - _dice(torch.sigmoid(lg) * v, y * v).mean()
    raise ValueError(name)
