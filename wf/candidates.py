"""Candidate space for the architecture search.

Measured throughput (L4, bf16, channels_last, bs 256) decides what is worth
screening: SMP encoders run at 2k-6k samples/s while full-resolution dilated
trunks at 250-300, so the search is centred on encoder-decoder models.
"""

from __future__ import annotations

import copy

PROTOCOLS = {
    1: {"n_train": 50000, "epochs": 12, "lr": 3e-3, "batch_size": 384, "warmup_epochs": 1,
        "max_seconds": 900, "patience": 99, "abort_below": 0.33, "abort_after": 4, "eval_n": 5000},
    2: {"n_train": 100000, "epochs": 30, "lr": 2.5e-3, "batch_size": 384, "warmup_epochs": 2,
        "patience": 10},
    3: {"n_train": None, "epochs": 36, "lr": 2.2e-3, "batch_size": 384, "warmup_epochs": 2, "patience": 8},
}

BASE = {
    "data": {"root": "data/processed", "prev_dropout": 0.2},
    "loss": {"name": "bce_dice", "pos_weight": 5.0, "dice_weight": 1.0},
    "train": {"weight_decay": 1e-4, "amp": "bf16", "grad_clip": 1.0, "workers": 10, "eval_workers": 4,
              "ema_decay": 0.999},
}


def smp(arch: str, encoder: str, **kw) -> dict:
    return {"model": {"smp": {"arch": arch, "encoder": encoder, **kw}}}


def smp_up(arch: str, encoder: str, up: int, **kw) -> dict:
    return {"model": {**smp(arch, encoder, **kw)["model"], "up": up}}


CANDIDATES: dict[str, dict] = {
    # --- encoder-decoder with pretrained backbones (the fast lane) ---
    "smp_unet_r18": smp("Unet", "resnet18"),
    "smp_unet_r34": smp("Unet", "resnet34"),
    "smp_unet_r50": smp("Unet", "resnet50"),
    "smp_unetpp_r18": smp("UnetPlusPlus", "resnet18"),
    "smp_unetpp_r34": smp("UnetPlusPlus", "resnet34"),
    "smp_deeplab_r18": smp("DeepLabV3Plus", "resnet18"),
    "smp_deeplab_r34": smp("DeepLabV3Plus", "resnet34"),
    "smp_fpn_r18": smp("FPN", "resnet18"),
    "smp_fpn_r50": smp("FPN", "resnet50"),
    "smp_psp_r18": smp("PSPNet", "resnet18"),
    "smp_linknet_r34": smp("LinkNet", "resnet34"),
    "smp_manet_b0": smp("MAnet", "efficientnet-b0"),
    "smp_unet_effb1": smp("Unet", "efficientnet-b1"),
    "smp_unet_mitb0": smp("Unet", "mit_b0"),
    "smp_unet_mitb1": smp("Unet", "mit_b1"),
    "smp_unet_convnext_t": smp("Unet", "tu-convnext_tiny"),
    "smp_unet_r18_nopret": smp("Unet", "resnet18", weights=None),

    # --- same encoders on a 2x grid (tiny targets, see notes in SMPNet) ---
    "smp_unet_r18_up2": smp_up("Unet", "resnet18", 2),
    "smp_unet_r34_up2": smp_up("Unet", "resnet34", 2),

    # --- custom trunks (heavier per sample, but no ImageNet prior) ---
    "unet32_4l": {"model": {"trunk": "unet", "widths": [32, 64, 128, 256], "depths": [2, 2, 2, 2]}},
    "unet48": {"model": {"trunk": "unet", "widths": [48, 96, 192], "depths": [2, 2, 2]}},
    "unet48_lk": {"model": {"trunk": "unet", "widths": [48, 96, 192], "depths": [2, 2, 2], "block": "lk"}},
    "unet64": {"model": {"trunk": "unet", "widths": [64, 128, 256], "depths": [2, 2, 2]}},
    "hrnet32": {"model": {"trunk": "hrnet", "width": 32, "stages": 3}},
    "hrnet48": {"model": {"trunk": "hrnet", "width": 48, "stages": 3}},
    "dil64": {"model": {"trunk": "fullres", "width": 64,
                        "blocks": [[1, 2], [2, 2], [4, 2], [8, 2], [16, 2], [8, 2], [4, 2]]}},

    # --- dual-branch fuel/weather encoder with a CAFIM gate per scale ---
    "dual32": {"model": {"trunk": "dual", "widths": [32, 64, 128], "depths": [2, 2, 2]}},
    "dual48": {"model": {"trunk": "dual", "widths": [48, 96, 192], "depths": [2, 2, 2]}},
    "dual32_deep": {"model": {"trunk": "dual", "widths": [32, 64, 128], "depths": [3, 3, 3]}},
    "dual48_bg": {"model": {"trunk": "dual", "widths": [48, 96, 192], "depths": [2, 2, 2],
                            "heads": ["main", "boundary", "growth"]},
                  "loss": {"name": "bce_dice", "pos_weight": 5.0, "boundary_weight": 0.3, "growth_weight": 0.3}},

    # --- multi-head (main + boundary / growth) ---
    "unet48_bg": {"model": {"trunk": "unet", "widths": [48, 96, 192], "depths": [2, 2, 2],
                            "heads": ["main", "boundary", "growth"]},
                  "loss": {"name": "bce_dice", "pos_weight": 5.0, "boundary_weight": 0.3, "growth_weight": 0.3}},
    "smp_unetpp_r18_bg": {"model": {"smp": {"arch": "UnetPlusPlus", "encoder": "resnet18"},
                                    "heads": ["main", "boundary"]},
                          "loss": {"name": "bce_dice", "pos_weight": 5.0, "boundary_weight": 0.3}},

    # --- loss / augmentation ablations on the base encoder ---
    "smp_unet_r18_tversky": {**smp("Unet", "resnet18"), "loss": {"name": "tversky", "alpha": 0.3, "beta": 0.7}},
    "smp_unet_r18_ftl": {**smp("Unet", "resnet18"),
                         "loss": {"name": "focal_tversky", "alpha": 0.4, "beta": 0.6, "gamma": 1.33}},
    "smp_unet_r18_lovasz": {**smp("Unet", "resnet18"), "loss": {"name": "lovasz"}},
    "smp_unet_r18_drop4": {**smp("Unet", "resnet18"), "data": {"root": "data/processed", "prev_dropout": 0.4}},
    "smp_unet_r18_drop0": {**smp("Unet", "resnet18"), "data": {"root": "data/processed", "prev_dropout": 0.0}},
}

# Cheapest-first-ish, likely winners on top: the deadline may cut the tail.
ORDER = [
    "smp_unet_r18", "smp_unet_r34", "smp_deeplab_r18", "smp_unetpp_r18", "smp_unet_mitb0",
    "dual32", "dual48", "smp_fpn_r18", "smp_manet_b0", "unet32_4l", "smp_unet_r50",
    "smp_unet_convnext_t", "smp_unetpp_r34", "smp_deeplab_r34", "dual32_deep", "unet48",
    "hrnet32", "dual48_bg", "smp_psp_r18", "smp_linknet_r34", "smp_unet_effb1",
    "smp_unet_r18_drop4", "smp_unet_r18_tversky", "smp_unet_r18_lovasz", "smp_unet_r18_ftl",
    "smp_unetpp_r18_bg", "smp_unet_r18_nopret", "unet48_bg", "unet48_lk", "unet64",
    "hrnet48", "dil64", "smp_unet_mitb1", "smp_fpn_r50",
]


def make_cfg(name: str, stage: int = 1) -> dict:
    cfg = copy.deepcopy(BASE)
    cand = copy.deepcopy(CANDIDATES[name])
    cfg["name"] = name
    for k, v in cand.items():
        cfg.setdefault(k, {}).update(v)
    cfg["train"].update(PROTOCOLS[stage])
    return cfg
