# Architecture

One run, start to finish. Mermaid source: [`architecture.mmd`](architecture.mmd) ·
renders: [`architecture.svg`](architecture.svg) · [`architecture.png`](architecture.png)

Custom trunk in detail: [`td_architecture.mmd`](td_architecture.mmd) — renders
[`.svg`](td_architecture.svg) · [`.png`](td_architecture.png) (also embedded below).

```mermaid
flowchart TB
  subgraph DATA["Data — 64×64 patches · 23 channels"]
    F["features.f16 (memmap)<br/>ndvi · terrain · weather<br/>prev_fire · fire_dist"]
    C["fire_class.u8"]
    P["prev_class.u8"]
    T["y = FireMask ≥ 7<br/>valid = not in {4, 6}"]
    A["dihedral ×8 augmentation<br/>u/v rotated by Jacobian<br/>prev_fire dropout 0.2"]
    C --> T
    F --> A
    T --> A
    P --> A
  end

  subgraph MODEL["Model — 64×64 in / 64×64 out"]
    N["normalise<br/>per-channel mean / std"]
    E["encoder<br/>ResNet-18 / ResNet-34 / MiT-B0<br/>ImageNet weights<br/>or TD orthogonal-transform branches"]
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

  subgraph ENS["Ensemble — 2 checkpoints"]
    W["smp_unetpp_r18 0.6<br/>td_gate3 0.4"]
    TT["dihedral TTA-4"]
    W --> TT
  end

  S --> W
  TT --> TEST["test split · 12,287 patches · evaluated once<br/>macro Dice 0.4826"]

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

## TD trunk — the custom architecture

`trunk: "td"` (candidate `td_gate3`). Same input/output contract as the U-Net line —
23×64×64 in, 1×64×64 logits out — but the encoder runs one orthogonal-transform
branch per basis and fuses them at every scale, so no single branch commits alone.

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 35, "rankSpacing": 45}}}%%
flowchart TB
  X["input x<br/>(B, 23, 64, 64)"] --> N["normalise<br/>(x - mean) / std"]
  N --> BR

  subgraph BR["3 parallel encoders · BranchEnc(cin=23, width=16, depth=1)"]
    direction LR
    H["ht · Walsh–Hadamard"]
    D["dct · DCT-II"]
    C["conv · no transform"]
  end

  BR --> FY["feature pyramid per branch<br/>f0 16@64 · f1 32@32 · f2 64@16"]
  FY --> FU{"Fuse at every scale<br/>k = 3 branches"}

  FU --> G["gate · per-pixel softmax<br/>out = Σ a_i · proj_i(f_i)"]
  FU --> S2["sum · per-branch scalar<br/>out = Σ w_i · f_i"]

  G --> U2
  S2 -.-> U2

  subgraph DEC["shared decoder and head"]
    direction TB
    U2["up 64→32<br/>bilinear 16 → 32"] --> M1["concat f1 (32)<br/>cna 64→32"]
    M1 --> U1["up 32→16<br/>bilinear 32 → 64"] --> M0["concat f0 (16)<br/>cna 32→16"]
  end

  DEC --> HD["Head · cna 16→16 + 1×1"] --> SG["sigmoid<br/>P(fire) · (B, 1, 64, 64)"]

  %% ---- inner blocks, anchored next to the stage they belong to ----

  BR ~~~ SPEC
  subgraph SPEC["SpecBlock · one per stage (repeated depth times)"]
    direction TB
    BODY["cna(c,c) ×2<br/>3×3 conv · BN · GELU"] --> K{"branch kind"}
    K -->|"ht / dct"| SP["Spectral<br/>z = B x Bᵀ<br/>z = z · gain<br/>z = sign(z)·ReLU(|z| − softplus(thr))<br/>out = Bᵀ z B"]
    K -->|"conv"| ID["identity"]
    SP --> R["residual: x + out"]
    ID --> R
  end

  FU ~~~ GATE
  subgraph GATE["Fuse detail (gate)"]
    direction TB
    P1["per branch: conv1×1 proj"] --> ST["stack k feature maps"]
    CA["cat k feature maps"] --> G2["conv3×3 c → GELU → conv1×1 k"] --> SM["softmax over k"]
    ST --> OUT["weighted sum<br/>(B, c, H, W)"]
    SM --> OUT
  end

  classDef io fill:#eef6ff,stroke:#4a7fb5,color:#123
  classDef op fill:#f4ecff,stroke:#7a4fb5,color:#123
  classDef alt fill:#f6f6f6,stroke:#888,color:#333,stroke-dasharray: 4 3
  class X,N,SG io
  class H,D,C,FY,FU,G,U2,M1,U1,M0,HD,BODY,SP,G2,SM,OUT op
  class S2,ID alt
```

`cna` = Conv3×3 (no bias) → BatchNorm → GELU. `B` is a fixed orthonormal matrix
(Walsh–Hadamard, or DCT-II) baked in as a buffer; `gain` and `thr` are learned
per coefficient, so the transform itself never has to be learned.

### Shape and parameter budget (`td_gate3`)

| stage | tensor | parameters |
|---|---|---|
| branch `ht` (stem + 3 stages) | 23 → 16@64 → 32@32 → 64@16 | 353k |
| branch `dct` (stem + 3 stages) | 23 → 16@64 → 32@32 → 64@16 | 353k |
| branch `conv` (control, same stem/stages) | 23 → 16@64 → 32@32 → 64@16 | 124k |
| fuse ×3 scales (gate) | 16/32/64 channels | 162k |
| decoder + head | 64→32→16 → 1 logit | 49k |
| **total** | | **1.04M** |

### Why this shape

| Choice | Reason |
|---|---|
| Deepest feature is 16×16, not 2×2 | A 32× downsampled trunk decides "is this 500-px region burning" on a 2×2 grid; the TD trunk keeps the coarse decision at 16×16 and the skip path at the native 64×64. |
| Two spectral bases + one plain conv | `ht` gives axis-aligned cheap sparsity, `dct` gives smooth energy compaction, `conv` keeps a path for features neither basis represents. With one branch the model is just a small U-Net. |
| Gate fusion, not sum | Ties sum on score (0.4580 vs 0.4578 screen) but lets each scale pick a branch; sum stays available as `fusion: "sum"`. |
| Width 16, depth 1 | Widening (24) or deepening (2) bought nothing at 6× the training time. |
| 1M params | Ties the 16M ResNet-18 U-Net++ (0.4662 vs 0.4681 eval), which is the parameter-efficiency claim of the TD-FusionUNet line that actually reproduced. |
| No pretrained weights | Forces the transform branches to carry the representation; also makes the whole run 1M params on disk. |

Known failure mode, measured on test: the trunk predicts shape at roughly 1-km
resolution, so large fires come out as a crescent on the burn-scar rim rather than a
filled area. Parameter count and transform choice do not move that
(`docs/RESULTS.md`, *What did not work*).

## Why these pieces

| Piece | Reason it survived |
|---|---|
| U-Net++ / U-Net decoder on ImageNet trunk | Best of six screened families (0.4681 eval, rebuilt cache) |
| `pos_weight 5` BCE + soft Dice | Dice controls the empty-ish patches, BCE keeps gradients sane |
| EMA + bf16 + channels_last | Free accuracy and ~1.6× throughput on L4 |
| Dihedral ×8, u/v Jacobian-rotated | Only augmentation that is physically consistent |
| 2-model blend, weights on eval | +0.002 over the best single; the gap is inside the CI, so it is insurance, not a claim |
| TD orthogonal-transform trunk (1M params) | Ties the 16M U-Net++ at 6% of the parameters; kept in the blend for diversity |

## Rejected after measurement

| Idea | Why it lost |
|---|---|
| DeepLabV3+ (stride-4 output) | −0.10 macro Dice, too coarse for 7-px fires |
| 2× input upsampling | 4× compute, no gain |
| Dual-branch fuel/weather + CAFIM | Too slow to converge inside the budget |
| Learned pixel-gate fusion in the TD trunk | No better than a scalar branch sum |
| Multi-head (boundary + growth) TD variant | 30% slower, no gain |
| Per-sample thresholds, prediction dilation | No honest gain on eval; see `docs/RESULTS.md` |
| Shift TTA, 8-way TTA, centre-radius mask, extra seeds | Neutral or negative; see `docs/RESULTS.md` |
