# Architecture

One run, start to finish. Mermaid source: [`architecture.mmd`](architecture.mmd) ·
renders: [`architecture.svg`](architecture.svg) · [`architecture.png`](architecture.png)

```mermaid
flowchart TB
  subgraph DATA["Data — 64×64 patches · 23 channels"]
    F["features.npy (memmap)<br/>ndvi · terrain · weather<br/>prev_fire · fire_dist"]
    C["fire_class.npy"]
    P["prev_class.npy"]
    T["y = FireMask ≥ 7<br/>valid = not in {4, 6}"]
    A["dihedral ×8 augmentation<br/>u/v rotated by Jacobian<br/>prev_fire dropout 0.2"]
    C --> T
    F --> A
    T --> A
    P --> A
  end

  subgraph MODEL["Model — 64×64 in / 64×64 out"]
    N["normalise<br/>per-channel mean / std"]
    E["encoder<br/>ResNet-18 / ResNet-34 / MiT-B0<br/>ImageNet weights"]
    D["decoder<br/>U-Net · U-Net++"]
    H["1×1 head<br/>1 logit map"]
    N --> E --> D --> H
  end

  A --> N

  subgraph TRAIN["Training"]
    L["loss = BCEWithLogits<br/>pos_weight 5, masked by valid<br/>+ soft Dice"]
    O["AdamW · cosine · 1–2 warmup epochs<br/>bf16 · channels_last · EMA"]
    L --> O
  end

  H --> L
  O --> V["eval every epoch<br/>macro Dice @ 0.50 on 10,512 patches"]

  V -->|"stage 1: below 0.33 at epoch 4"| X["abort candidate"]
  V -->|"finalist: best checkpoint"| S["threshold sweep 0.30–0.85<br/>eval only"]
  S --> M["metrics<br/>macro / micro Dice · IoU · P / R<br/>PR-AUC · boundary F1 · hard Dice"]

  subgraph ENS["Ensemble — 5 checkpoints"]
    W["smp_unetpp_r18 0.25<br/>smp_unet_mitb0 seeds 0/1/2 ×0.20<br/>smp_unet_r34 0.15"]
    TT["dihedral TTA-4"]
    W --> TT
  end

  S --> W
  TT --> TEST["test split · 12,467 patches · evaluated once<br/>macro Dice 0.4823"]

  classDef data fill:#eef6ff,stroke:#4a7fb5,color:#123
  classDef model fill:#f4ecff,stroke:#7a4fb5,color:#123
  classDef train fill:#fff4e6,stroke:#b57b2a,color:#123
  classDef eval fill:#e9f7ef,stroke:#2f8f5b,color:#123
  classDef rej fill:#fdecec,stroke:#b54a4a,color:#512
  class F,C,P,T,A data
  class N,E,D,H model
  class L,O train
  class V,S,M,W,TT,TEST eval
  class X rej
```

## Shape flow

`(B, 23, 64, 64)` → encoder ×32 down → `(B, 512, 2, 2)` → decoder + skip connections →
`(B, 1, 64, 64)` logits → sigmoid at the tuned threshold. Predictions are made on the
64×64 patch grid, i.e. the native label resolution; upsampling at inference was
measured and gave nothing (`docs/RESULTS.md`).

## Why these pieces

| Piece | Reason it survived |
|---|---|
| U-Net++ / U-Net decoder on ImageNet trunk | Best of six screened families (0.4686 eval) |
| `pos_weight 5` BCE + soft Dice | Dice controls the empty-ish patches, BCE keeps gradients sane |
| EMA + bf16 + channels_last | Free accuracy and ~1.6× throughput on L4 |
| Dihedral ×8, u/v Jacobian-rotated | Only augmentation that is physically consistent |
| 5-model blend, weights on eval | +0.002–0.003 over best single; the gap is inside the CI, so it is insurance, not a claim |

## Rejected after measurement

| Idea | Why it lost |
|---|---|
| DeepLabV3+ (stride-4 output) | −0.10 macro Dice, too coarse for 7-px fires |
| 2× input upsampling | 4× compute, no gain |
| Dual-branch fuel/weather + CAFIM | Too slow to converge inside the budget |
| Shift TTA, 8-way TTA, centre-radius mask, extra seeds | Neutral or negative; see `docs/RESULTS.md` |
