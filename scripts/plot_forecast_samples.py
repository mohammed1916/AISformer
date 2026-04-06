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

from utils import haversine

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
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for forecast trajectories.')
    parser.add_argument('--top-k', type=int, default=10, help='Top-k filtering for sampling logits.')
    parser.add_argument('--max-sample-attempts', type=int, default=5, help='Maximum resampling attempts for stable future paths.')
    parser.add_argument('--reject-first-km', type=float, default=20.0, help='Reject samples with first-step distance above this km.')
    parser.add_argument('--reject-step-km', type=float, default=20.0, help='Reject samples with a single step jump above this km.')
    return parser.parse_args()


def sample_safe_future_paths(
    model,
    seqs,
    token_types,
    valid_mask,
    port_features,
    land_features,
    cfg,
    forecast_len,
    n_samples,
    temperature,
    top_k,
    max_attempts,
    reject_first_km,
    reject_step_km,
):
    samples = []
    prev_real = infer_future.to_real_points(seqs[0, :10, :4].cpu().numpy(), cfg, 'normalized') if seqs.shape[1] >= 10 else None
    prev_point = prev_real[-1:] if prev_real is not None else None
    prev_rad = None
    if prev_point is not None:
        prev_rad = torch.tensor(prev_point, dtype=torch.float32) * torch.pi / 180.0

    for sample_idx in range(n_samples):
        accepted = None
        best = None
        best_score = float('inf')
        for attempt in range(max_attempts):
            completed = trainers.predict_gap(
                model,
                seqs,
                token_types,
                valid_mask,
                port_context=port_features,
                land_context=land_features,
                sample=True,
                temperature=temperature,
                top_k=top_k,
            )
            future_norm = completed[0, len(prev_real) : len(prev_real) + forecast_len].detach().cpu().numpy()
            future_real = infer_future.denormalize_points(future_norm, cfg)
            future_rad = torch.tensor(future_real[:, :2], dtype=torch.float32) * torch.pi / 180.0
            d_from_prev = haversine(prev_rad.unsqueeze(0), future_rad.unsqueeze(0)).squeeze(0).cpu().numpy()
            d_steps = (haversine(future_rad[:-1].unsqueeze(0), future_rad[1:].unsqueeze(0)).squeeze(0).cpu().numpy() if future_rad.shape[0] > 1 else np.array([]))
            first_dist = float(d_from_prev[0])
            max_jump = float(np.max(d_steps)) if d_steps.size > 0 else 0.0
            score = first_dist + max_jump
            if first_dist <= reject_first_km and max_jump <= reject_step_km:
                accepted = future_real
                break
            if score < best_score:
                best_score = score
                best = future_real
        samples.append(accepted if accepted is not None else best)
    return samples


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
            config=cfg,
        )

        seqs = seqs.to(cfg.device)
        token_types = token_types.to(cfg.device)
        valid_mask = valid_mask.to(cfg.device)
        port_features = port_features.to(cfg.device)
        land_features = land_features.to(cfg.device)

        samples = sample_safe_future_paths(
            model,
            seqs,
            token_types,
            valid_mask,
            port_features,
            land_features,
            cfg,
            args.forecast_len,
            args.n_samples,
            args.temperature,
            args.top_k,
            args.max_sample_attempts,
            args.reject_first_km,
            args.reject_step_km,
        )

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
