"""Primary model commands. All experiment computation runs on Kaggle."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .primary import REFERENCE_THRESHOLD, configuration, load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train', help='Train only WildfireTD4 with the selected recipe')
    tr.add_argument('--root', required=True, help='Cache with train/eval; prefer exploration/search_data')
    tr.add_argument('--out', default='/kaggle/working/primary_runs')
    tr.add_argument('--seed', type=int, default=0)
    ev = sub.add_parser('evaluate', help='Evaluate an existing checkpoint, without training')
    ev.add_argument('--root', required=True)
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--split', choices=['eval', 'confirm', 'test'], default='eval')
    ev.add_argument('--threshold', type=float, default=None)
    ev.add_argument('--out', required=True)
    pr = sub.add_parser('predict', help='Predict an unnormalized N x 33 x 64 x 64 .npy array')
    pr.add_argument('--input', required=True)
    pr.add_argument('--checkpoint', required=True)
    pr.add_argument('--threshold', type=float, default=None)
    pr.add_argument('--out', required=True)
    args = ap.parse_args()
    if not Path('/kaggle/working').is_dir() or not torch.cuda.is_available():
        raise RuntimeError('Run model computation inside Kaggle with a GPU enabled')
    if args.command == 'train':
        from .train import run
        run(configuration(args.root, args.seed), f'wildfire_td4_seed{args.seed}', args.out)
        return
    model, checkpoint = load_checkpoint(args.checkpoint, 'cuda')
    summary = Path(args.checkpoint).parent / 'summary.json'
    if args.threshold is not None:
        threshold = args.threshold
    elif summary.exists():
        threshold = json.loads(summary.read_text())['best_thr']
    elif 'thr' in checkpoint.get('metrics', {}):
        threshold = checkpoint['metrics']['thr']
    else:
        raise ValueError('No checkpoint threshold found; specify --threshold explicitly')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == 'evaluate':
        from .data import FireDataset
        from .metrics import evaluate
        from .train import _loader, predict
        ds = FireDataset(args.root, args.split, aux=False)
        p, y, v, _ = predict(model, _loader(ds, 64, False, 2), torch.device('cuda'), torch.float16)
        report = {'checkpoint': str(args.checkpoint), 'split': args.split, 'threshold': threshold,
                  'tta': 1, 'metrics': evaluate(p, y, v, threshold)}
        (out / 'metrics.json').write_text(json.dumps(report, indent=2))
        np.save(out / 'probability.npy', p.astype(np.float16))
        print(json.dumps(report, indent=2))
    else:
        x = np.load(args.input, mmap_mode='r')
        if x.ndim != 4 or x.shape[1:] != (33, 64, 64) or not len(x):
            raise ValueError('Input must be a nonempty N x 33 x 64 x 64 array')
        p = np.lib.format.open_memmap(out / 'probability.npy', mode='w+', dtype=np.float16,
                                     shape=(len(x), 64, 64))
        mask = np.lib.format.open_memmap(out / 'mask.npy', mode='w+', dtype=np.uint8,
                                        shape=p.shape)
        with torch.inference_mode():
            for start in range(0, len(x), 64):
                batch = torch.from_numpy(np.array(x[start:start+64], dtype=np.float32)).cuda()
                if not torch.isfinite(batch).all():
                    raise ValueError('Input contains nonfinite values')
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = model(batch)['main']
                values = pred.float().sigmoid().cpu().numpy()[:, 0]
                p[start:start+len(values)] = values
                mask[start:start+len(values)] = values > threshold
        p.flush()
        mask.flush()
        (out / 'prediction.json').write_text(json.dumps({'checkpoint': args.checkpoint,
                                                       'threshold': threshold, 'tta': 1, 'rows': len(x)}, indent=2))


if __name__ == '__main__':
    main()
