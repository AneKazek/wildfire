"""Candidate space for the architecture search.

Measured throughput (L4, bf16, channels_last, bs 256) decides what is worth
screening: SMP encoders run at 2k-6k samples/s while full-resolution dilated
trunks at 250-300, so the search is centred on encoder-decoder models.
"""

from __future__ import annotations

import copy

PROTOCOLS = {
    1: {"n_train": 50000, "epochs": 30, "lr": 3e-3, "batch_size": 384, "warmup_epochs": 2,
        "max_seconds": 1500, "patience": 99, "eval_n": 5000},
    2: {"n_train": 100000, "epochs": 30, "lr": 2.5e-3, "batch_size": 384, "warmup_epochs": 2,
        "patience": 10},
    3: {"n_train": None, "epochs": 36, "lr": 2.2e-3, "batch_size": 384, "warmup_epochs": 2, "patience": 8},
}

BASE = {
    "data": {"root": "data/tsf", "prev_dropout": 0.2},
    "loss": {"name": "bce_dice", "pos_weight": 5.0, "dice_weight": 1.0},
    "train": {"weight_decay": 1e-4, "amp": "bf16", "grad_clip": 1.0, "workers": 10, "eval_workers": 4,
              "ema_decay": 0.999},
}


def smp(arch: str, encoder: str, **kw) -> dict:
    return {"model": {"smp": {"arch": arch, "encoder": encoder, **kw}}}


def smp_up(arch: str, encoder: str, up: int, **kw) -> dict:
    return {"model": {**smp(arch, encoder, **kw)["model"], "up": up}}


def pv(prep: str, drop: str = "") -> dict:
    """Same trunk as the winner, only the input preprocessing changes."""
    return {**smp("UnetPlusPlus", "resnet18"),
            "data": {"root": "data/tsf", "prev_dropout": 0.2, "prep": prep, "drop": drop}}


def td(**kw) -> dict:
    """Orthogonal-transform trunk (TD-FusionUNet line)."""
    return {"model": {"trunk": "td", **kw}}


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
    "smp_unet_r18_drop4": {**smp("Unet", "resnet18"), "data": {"root": "data/tsf", "prev_dropout": 0.4}},
    "smp_unet_r18_drop0": {**smp("Unet", "resnet18"), "data": {"root": "data/tsf", "prev_dropout": 0.0}},

    # --- preprocessing sweep: identical trunk, different input channels ---
    "pp_lean": pv("base", "lean"),
    "pp_ctx": pv("ctx"),
    "pp_scar": pv("scar"),
    "pp_wind": pv("wind"),
    "pp_scarwind": pv("scarwind"),
    "pp_all": pv("all"),
    "pp_all_lean": pv("all", "lean"),
    "pp_dist": pv("dist", "nodist"),

    # --- orthogonal-transform trunks (TD-FusionUNet line) ---
    "td_sum": td(branches=["ht", "dct"], fusion="sum", width=16),
    "td_gate": td(branches=["ht", "dct"], fusion="gate", width=16),
    "td_gate3": td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
    "td_gate_deep": td(branches=["ht", "dct"], fusion="gate", width=16, depth=2),
    "td_gate_wide": td(branches=["ht", "dct"], fusion="gate", width=24),
    "td_gate_mh": {**td(branches=["ht", "dct"], fusion="gate", width=16,
                        heads=["main", "boundary", "growth"]),
                   "loss": {"name": "bce_dice", "pos_weight": 5.0, "boundary_weight": 0.3, "growth_weight": 0.3}},
    "td_gate_pf": {**td(branches=["ht", "dct"], fusion="gate", width=16),
                   "data": {"root": "data/tsf", "prev_dropout": 0.2, "pf_aug": "both"}},

    # --- round 2: soft labels / scar-channel ablations / physics channels ---
    "td_gate3_soft8": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                       "data": {"root": "data/tsf", "prev_dropout": 0.2, "soft": 0.8}},
    "td_gate3_soft15": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                        "data": {"root": "data/tsf", "prev_dropout": 0.2, "soft": 1.5}},
    "td_gate3_noscar": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                        "data": {"root": "data/tsf", "prev_dropout": 0.2, "drop": "noscar"}},
    "td_gate3_scarwind": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                          "data": {"root": "data/tsf", "prev_dropout": 0.2, "prep": "scarwind"}},
    "td_gate3_ftl": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                     "loss": {"name": "focal_tversky", "alpha": 0.4, "beta": 0.6, "gamma": 1.33}},
    "td_gate3_pf": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                    "data": {"root": "data/tsf", "prev_dropout": 0.2, "pf_aug": "both"}},

    # --- round 3: precision-weighted objective (FP costs more than FN) ---
    "td_ptv": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
               "loss": {"name": "focal_tversky", "alpha": 0.7, "beta": 0.3, "gamma": 1.33}},

    # --- round 3: moment-matching term (where the mass sits / how big it is) ---
    "td_mom05": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                 "loss": {"name": "bce_dice", "pos_weight": 5.0, "moment_weight": 0.5}},
    "td_mom10": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                 "loss": {"name": "bce_dice", "pos_weight": 5.0, "moment_weight": 1.0}},

    # --- round 4: architecture brute force around the same trunk ---
    "td_attn": td(branches=["ht", "dct", "conv"], fusion="gate", width=16, attn=True),
    "td_dec2": td(branches=["ht", "dct", "conv"], fusion="gate", width=16, dec=2),
    "td_4s": td(branches=["ht", "dct", "conv"], fusion="gate", width=16, sizes=(64, 32, 16, 8)),
    "td_4s_dec2": td(branches=["ht", "dct", "conv"], fusion="gate", width=16,
                     sizes=(64, 32, 16, 8), dec=2),
    "td_w8": td(branches=["ht", "dct", "conv"], fusion="gate", width=8),
    "td_d3": td(branches=["ht", "dct", "conv"], fusion="gate", width=16, depth=3),

    # --- round 5: explicit area head + top-k decode (the measured bottleneck) ---
    "td_area": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16, area=32),
                "loss": {"name": "bce_dice", "pos_weight": 5.0, "area_weight": 1.0}},
    "td_area_w2": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16, area=64),
                   "loss": {"name": "bce_dice", "pos_weight": 5.0, "area_weight": 2.0}},

    # --- round 6: optimise the per-image metric directly ---
    "td_lovasz": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16),
                  "loss": {"name": "lovasz"}},

    # --- round 7: deep supervision and the high-resolution sub-pixel branch ---
    "td_ds": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16, ds=True),
              "loss": {"name": "bce_dice", "pos_weight": 5.0, "ds_weight": 0.5}},
    "td_hr": td(branches=["ht", "dct", "conv"], fusion="gate", width=16, hr=8),
    "td_hr_ds": {**td(branches=["ht", "dct", "conv"], fusion="gate", width=16, hr=8, ds=True),
                 "loss": {"name": "bce_dice", "pos_weight": 5.0, "ds_weight": 0.5}},
}

# Cheapest-first-ish, likely winners on top: the deadline may cut the tail.
ORDER = [
    "td_sum", "td_gate", "td_gate3", "td_gate_deep", "td_gate_wide", "td_gate_mh", "td_gate_pf",
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
