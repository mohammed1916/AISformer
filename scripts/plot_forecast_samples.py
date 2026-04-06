#!/usr/bin/env python
# coding=utf-8

"""Generate a PNG showing multiple sampled forecast trajectories."""

from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_forecast import infer_future
from src_interpolation import models, trainers
import src_interpolation.land_context as land_context
import src_interpolation.port_context as port_context


def parse_args():
    parser = argparse.ArgumentParser(description='Generate forecast sample plots for multiple test tracks.')
    parser.add_argument('--n-tracks', type=int, default=3, help='Number of test tracks to visualize.')
    parser.add_argument('--n-samples', type=int, default=3, help='Number of sampled futures per track.')
    parser.add_argument('--forecast-len', type=int, default=10, help='Number of future points to generate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for forecast trajectories.')
    parser.add_argument('--top-k', type=int, default=20, help='Top-k filtering for sampling logits.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = infer_future.Config()
    test_path = ROOT / 'data' / 'ct_dma' / 'ct_dma_test.pkl'
    with test_path.open('rb') as f:
        data = pickle.load(f)

    n_tracks = min(args.n_tracks, len(data))
    if n_tracks == 0:
        raise ValueError('No test tracks found in the dataset.')

    checkpoint = Path(cfg.ckpt_path)
    model = models.TrAISformerInterpolation(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    fig, axes = plt.subplots(n_tracks, 1, figsize=(6, 4 * n_tracks), squeeze=False)
    for track_idx in range(n_tracks):
        track = data[track_idx]
        prev_points = np.asarray(track['traj'][:10, :4], dtype=np.float32)
        prev_real = infer_future.to_real_points(prev_points, cfg, 'normalized')
        prev_norm = prev_points

        port_encoder = None
        land_encoder = None
        if getattr(cfg, 'use_port_context', False):
            port_encoder = port_context.PortContextEncoder.from_config(cfg)
        if getattr(cfg, 'use_land_context', False):
            land_encoder = land_context.LandContextEncoder.from_config(cfg)

        seqs, token_types, valid_mask, target_mask, port_features, land_features = infer_future.datasets.build_forecast_sequence(
            prev_norm,
            args.forecast_len,
            cfg.max_seqlen,
            port_encoder=port_encoder,
            land_encoder=land_encoder,
            prev_real_points=prev_real[:, :2],
            port_context_size=getattr(cfg, 'port_context_size', 0),
            land_context_size=getattr(cfg, 'land_context_size', 0),
        )

        seqs = seqs.to(cfg.device)
        token_types = token_types.to(cfg.device)
        valid_mask = valid_mask.to(cfg.device)
        port_features = port_features.to(cfg.device)
        land_features = land_features.to(cfg.device)

        samples = []
        for _ in range(args.n_samples):
            completed = trainers.predict_gap(
                model,
                seqs,
                token_types,
                valid_mask,
                port_context=port_features,
                land_context=land_features,
                sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            future_norm = completed[0, len(prev_norm) : len(prev_norm) + args.forecast_len].detach().cpu().numpy()
            samples.append(infer_future.denormalize_points(future_norm, cfg))

        ax = axes[track_idx][0]
        ax.plot(prev_real[:, 1], prev_real[:, 0], '-k', marker='x', label='history')
        for sample_idx, future_real in enumerate(samples):
            ax.plot(future_real[:, 1], future_real[:, 0], '-o', label=f'sample {sample_idx+1}')
        ax.set_title(f'Track {track_idx + 1}')
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.grid(True)
        ax.legend(loc='best')

    plt.tight_layout()
    out_path = ROOT / 'results_forecast' / f'forecast_sample_paths_{n_tracks}tracks_{args.n_samples}samples.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved forecast sample PNG: {out_path}')


if __name__ == '__main__':
    main()
