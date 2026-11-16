"""Matched baseline / external-weather late-fusion experiment (eval only)."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_ext
from .train import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='data/tsf')
    ap.add_argument('--out', default='results/latefusion')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--n-train', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    for split in ('train', 'eval'):
        ext = load_ext(args.root, split)
        if ext is None or ext.shape[1] != 8 or not np.isfinite(ext).all():
            raise ValueError(f'{split}: build valid eight-column ext.npy with wf.ext first')
    results = {}
    for name, late in [('baseline', 0), ('weather_late', 8)]:
        cfg = {
            'seed': args.seed,
            'data': {'root': args.root, 'prev_dropout': 0.2, 'in_memory': False},
            'model': {'trunk': 'td', 'branches': ['ht', 'dct', 'conv'],
                      'width': 16, 'sizes': [64, 32, 16, 8], 'late': late},
            'loss': {'name': 'bce_dice', 'pos_weight': 5.0},
            'train': {'epochs': args.epochs, 'batch_size': args.batch_size,
                      'eval_batch_size': 64, 'n_train': args.n_train or None,
                      'workers': 2, 'eval_workers': 2, 'amp': 'fp16',
                      'lr': 0.002, 'patience': args.epochs},
        }
        results[name] = run(cfg, f'{name}_seed{args.seed}', args.out)
    report = {
        'seed': args.seed, 'selection_split': 'eval', 'test_evaluated': False,
        'eval': {k: v['val_at_best_thr'] for k, v in results.items()},
        'thresholds': {k: v['best_thr'] for k, v in results.items()},
    }
    report['delta_macro_dice'] = (report['eval']['weather_late']['macro_dice'] -
                                  report['eval']['baseline']['macro_dice'])
    Path(args.out, 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
