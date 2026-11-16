"""Kaggle-only architecture/loss/fusion search with grouped confirmation events."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from .data import FireDataset, cache, load_ext
from .final import predict_tta
from .metrics import dice_per_sample, evaluate, sweep
from .train import _loader, build, run


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def grouped_root(source: Path, dest: Path):
    """Reserve whole eval events before any search; never read test."""
    if (dest / 'partition.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(source / 'eval/meta.csv', dtype={'roi': str})
    events = sorted(meta.roi.unique())
    if len(events) < 6:
        raise ValueError('Need at least six eval events for grouped confirmation')
    order = np.random.default_rng(2026).permutation(events)
    confirm = set(order[:max(3, len(events) // 3)])
    (dest / 'train').symlink_to((source / 'train').resolve(), target_is_directory=True)
    (dest / 'norm.json').write_text((source / 'norm.json').read_text())
    arrays = cache(source, 'eval')
    ext = load_ext(source, 'eval')
    for split, selected in [('eval', ~meta.roi.isin(confirm)), ('confirm', meta.roi.isin(confirm))]:
        d = dest / split
        d.mkdir(exist_ok=True)
        indices = np.flatnonzero(selected)
        for arr, name in zip(arrays, ['features.f16', 'fire_class.u8', 'prev_class.u8']):
            with (d / name).open('wb') as f:
                for start in range(0, len(indices), 16):
                    f.write(np.asarray(arr[indices[start:start + 16]]).tobytes())
        meta.iloc[indices].to_csv(d / 'meta.csv', index=False)
        if ext is not None:
            np.save(d / 'ext.npy', ext[indices])
    write_json(dest / 'partition.json', {'seed': 2026, 'confirm_events': sorted(confirm),
                                        'tuning_events': sorted(set(events) - confirm),
                                        'source': str(source), 'test_read': False})


def candidates(weather: bool):
    td = {'trunk': 'td', 'branches': ['ht', 'dct', 'conv'], 'width': 16,
          'sizes': [64, 32, 16, 8]}
    space = {
        'td4': {'model': td},
        'td4_attention': {'model': {**td, 'attn': True}},
        'td4_wide_decoder': {'model': {**td, 'dec': 2}},
        'td4_w24': {'model': {**td, 'width': 24}},
        'td4_d2': {'model': {**td, 'depth': 2}},
        'td4_deep_supervision': {'model': {**td, 'ds': True}, 'loss': {'ds_weight': 0.3}},
        'unet': {'model': {'trunk': 'unet', 'widths': [32, 64, 128, 256], 'depths': [2]*4}},
        'hrnet': {'model': {'trunk': 'hrnet', 'width': 24, 'stages': 3}},
        'dual_fuel_weather': {'model': {'trunk': 'dual', 'widths': [32, 64, 128], 'depths': [2]*3}},
    }
    for name, arch, encoder in [('unetpp_r18', 'UnetPlusPlus', 'resnet18'),
                                 ('unet_r34', 'Unet', 'resnet34'),
                                 ('unet_mitb0', 'Unet', 'mit_b0'),
                                 ('unet_mitb1', 'Unet', 'mit_b1'),
                                 ('unet_effb1', 'Unet', 'efficientnet-b1'),
                                 ('unet_convnext', 'Unet', 'tu-convnext_tiny')]:
        space[name] = {'model': {'smp': {'arch': arch, 'encoder': encoder, 'weights': 'imagenet'}}}
    if weather:
        space['td4_weather_scalar'] = {'model': {**td, 'late': 8}}
        space['td4_weather_spatial'] = {'model': {**td, 'late': 8, 'late_mode': 'spatial'}}
    return space


def configuration(root, patch, epochs, seed, n_train, max_seconds):
    cfg = {'seed': seed, 'data': {'root': str(root), 'in_memory': False, 'prev_dropout': 0.2},
           'model': {}, 'loss': {'name': 'bce_dice', 'pos_weight': 5., 'dice_weight': 1.},
           'train': {'epochs': epochs, 'n_train': n_train, 'batch_size': 32,
                     'eval_batch_size': 64, 'workers': 2, 'eval_workers': 2,
                     'amp': 'fp16', 'lr': 0.002, 'patience': 12, 'warmup_epochs': 2,
                     'checkpoint_sweep': True, 'max_seconds': max_seconds}}
    for section, values in patch.items():
        cfg[section].update(copy.deepcopy(values))
    return cfg


def isolated(cfg, name, out):
    """Isolate CUDA/OOM failures and resume completed runs without rerunning."""
    summary = out / name / 'summary.json'
    if summary.exists():
        return json.loads(summary.read_text())
    jobs = out / 'jobs'
    jobs.mkdir(exist_ok=True, parents=True)
    path = jobs / f'{name}.json'
    write_json(path, {'cfg': cfg, 'name': name, 'out': str(out)})
    with (jobs / f'{name}.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'wf.explore', '--job', str(path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    if code or not summary.exists():
        with (out / 'failures.jsonl').open('a') as f:
            f.write(json.dumps({'name': name, 'exit': code, 'log': str(jobs / f'{name}.log')}) + '\n')
        return None
    return json.loads(summary.read_text())


def prediction(summary, root, split, tta):
    cfg = summary['cfg']
    device = torch.device('cuda')
    model, prep, drop = build(cfg, root)
    model.load_state_dict(torch.load(summary['_checkpoint'], map_location='cpu', weights_only=False)['model'], strict=False)
    model = model.to(device).to(memory_format=torch.channels_last)
    ds = FireDataset(root, split, aux=False, prep=prep, drop=drop,
                     ext=load_ext(root, split) if cfg['model'].get('late') else None)
    prob, _ = predict_tta(model, _loader(ds, 64, False, 2), device, torch.float16, tta)
    y = np.asarray(ds.fire) >= 7
    valid = ~np.isin(np.asarray(ds.fire), (4, 6))
    del model
    torch.cuda.empty_cache()
    return prob, y, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job')
    ap.add_argument('--source', default='/kaggle/working/tsf_full')
    ap.add_argument('--out', default='/kaggle/working/exploration')
    ap.add_argument('--hours', type=float, default=8.)
    args = ap.parse_args()
    if args.job:
        job = json.loads(Path(args.job).read_text())
        run(job['cfg'], job['name'], job['out'])
        return
    if not Path('/kaggle/working').exists() or not torch.cuda.is_available():
        raise RuntimeError('Run this experiment inside Kaggle with GPU enabled')
    start = time.monotonic()
    deadline = start + args.hours * 3600
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    root = out / 'search_data'
    grouped_root(Path(args.source), root)
    weather = (root / 'train/ext.npy').exists()
    # Reject weather when geographic joins produced mostly missing observations.
    coverage = json.loads((Path(args.source) / 'weather_audit.json').read_text()) if (Path(args.source) / 'weather_audit.json').exists() else {}
    weather = weather and coverage.get('train_complete_fraction', 0) >= 0.1
    space = candidates(weather)
    successes = []

    def attempt(name, patch, epochs, seed=0, n_train=12000, seconds=900):
        if time.monotonic() + 180 > deadline:
            return None
        cfg = configuration(root, patch, epochs, seed, n_train, min(seconds, max(60, deadline-time.monotonic()-120)))
        summary = isolated(cfg, name, out / 'runs')
        if summary:
            summary['_checkpoint'] = str(out / 'runs' / name / 'best.pt')
            successes.append(summary)
            write_json(out / 'leaderboard.json', sorted(successes, key=lambda s: s['val_at_best_thr']['macro_dice'], reverse=True))
        return summary

    screened = []
    for name, patch in space.items():
        result = attempt('screen_' + name, patch, 18)
        if result:
            screened.append((name, patch, result))
    screened.sort(key=lambda r: r[2]['val_at_best_thr']['macro_dice'], reverse=True)
    if not screened:
        raise RuntimeError('All screening runs failed; see failures.jsonl and job logs')

    # Loss/regularization controls on the best architecture, not arbitrary new models.
    winner_name, winner_patch, _ = screened[0]
    variants = [('default', {}), ('bce_pw1', {'name': 'bce_dice', 'pos_weight': 1.}),
                ('tversky_recall', {'name': 'focal_tversky', 'alpha': .4, 'beta': .6, 'gamma': 1.33}),
                ('tversky_precision', {'name': 'focal_tversky', 'alpha': .7, 'beta': .3, 'gamma': 1.33}),
                ('lovasz', {'name': 'lovasz'})]
    promoted = [(n, p, r) for n, p, r in screened[:3]]
    for suffix, loss in variants[1:]:
        patch = copy.deepcopy(winner_patch)
        patch.setdefault('loss', {}).update(loss)
        result = attempt(f'loss_{winner_name}_{suffix}', patch, 18)
        if result:
            promoted.append((f'{winner_name}_{suffix}', patch, result))
    promoted.sort(key=lambda r: r[2]['val_at_best_thr']['macro_dice'], reverse=True)

    finalists = []
    # Fully train the best three recipes, then repeat their seeds if budget permits.
    for seed in (0, 1, 2):
        for name, patch, _ in promoted[:3]:
            result = attempt(f'final_{name}_seed{seed}', patch, 60, seed, None, 2400)
            if result:
                finalists.append(result)
    if not finalists:
        finalists = [r[2] for r in promoted[:3]]

    # Pick single model vs equal-weight diverse ensemble and TTA on tuning only.
    finalists.sort(key=lambda r: r['val_at_best_thr']['macro_dice'], reverse=True)
    finalists = finalists[:6]
    best = None
    for tta in (1, 4):
        probs = []
        for s in finalists:
            p, y, v = prediction(s, root, 'eval', tta)
            probs.append(p)
        for count in range(1, len(probs) + 1):
            blend = np.mean(probs[:count], axis=0)
            threshold, _ = sweep(blend, y, v, np.arange(.1, .91, .025))
            metrics = evaluate(blend, y, v, threshold, full=False)
            recipe = {'tta': tta, 'threshold': threshold, 'members': finalists[:count], 'tuning': metrics}
            if best is None or metrics['macro_dice'] > best['tuning']['macro_dice']:
                best = recipe
        del probs
    write_json(out / 'frozen_recipe.json', best)
    # Confirm exactly this recipe once on reserved events, with its fixed threshold.
    total = None
    for summary in best['members']:
        p, y, v = prediction(summary, root, 'confirm', best['tta'])
        total = p if total is None else total + p
    total /= len(best['members'])
    np.save(out / 'confirm_probability.npy', total.astype(np.float16))
    report = {'tuning': best['tuning'], 'confirmation': evaluate(total, y, v, best['threshold']),
              'threshold': best['threshold'], 'tta': best['tta'],
              'members': [s['name'] for s in best['members']], 'test_evaluated': False,
              'weather_enabled': weather, 'weather_audit': coverage,
              'minutes': (time.monotonic()-start)/60}
    meta = pd.read_csv(root / 'confirm/meta.csv', dtype={'roi': str})
    per_patch = dice_per_sample((total > best['threshold']) & v, y & v)
    event_scores = pd.Series(per_patch).groupby(meta.roi).mean()
    samples = np.random.default_rng(0).choice(event_scores.to_numpy(),
                                             (2000, len(event_scores)), replace=True).mean(1)
    report['confirmation_per_event'] = event_scores.to_dict()
    report['confirmation_event_macro'] = float(event_scores.mean())
    report['confirmation_event_bootstrap_95'] = np.percentile(samples, [2.5, 97.5]).tolist()
    write_json(out / 'report.json', report)
    print('EXPLORATION_REPORT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
