#!/usr/bin/env python
# coding=utf-8

"""Evaluate forecast divergence against the held-out test dataset."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src_forecast import infer_future
from src_interpolation import models, trainers
import src_interpolation.land_context as land_context
import src_interpolation.port_context as port_context
from utils import haversine


def parse_args():
    parser = argparse.ArgumentParser(description='Measure forecast divergence from previous trajectory on test tracks.')
    parser.add_argument('--n-tracks', type=int, default=10, help='Number of test tracks to evaluate.')
    parser.add_argument('--n-samples', type=int, default=3, help='Number of sampled futures per track.')
    parser.add_argument('--forecast-len', type=int, default=10, help='Number of future points to generate.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for generation.')
    parser.add_argument('--top-k', type=int, default=10, help='Top-k filtering for decoding.')
    parser.add_argument('--max-sample-attempts', type=int, default=5, help='Maximum resampling attempts for stable future paths.')
    parser.add_argument('--reject-first-km', type=float, default=20.0, help='Reject samples with first-step distance above this km.')
    parser.add_argument('--reject-step-km', type=float, default=20.0, help='Reject samples with a single step jump above this km.')
    parser.add_argument('--far-threshold', type=float, default=20.0, help='Threshold in km for far-off forecast from last observed point.')
    parser.add_argument('--jump-threshold', type=float, default=15.0, help='Threshold in km for abrupt jumps between consecutive predicted points.')
    return parser.parse_args()


def latlon_to_radians(coords, cfg):
    coords = torch.tensor(coords, dtype=torch.float32)
    coords = coords.clone()
    coords[..., 0] = coords[..., 0] * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
    coords[..., 1] = coords[..., 1] * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
    coords = coords * torch.pi / 180.0
    return coords


def sample_safe_future(
    model,
    seqs,
    token_types,
    valid_mask,
    port_features,
    land_features,
    cfg,
    forecast_len,
    temperature,
    top_k,
    max_attempts,
    reject_first_km,
    reject_step_km,
):
    prev_real = infer_future.to_real_points(seqs[0, :10, :4].cpu().numpy(), cfg, 'normalized')
    prev_point = prev_real[-1:]
    prev_rad = torch.tensor(prev_point, dtype=torch.float32) * torch.pi / 180.0

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
        future_norm = completed[0, 10:10 + forecast_len].detach().cpu().numpy()
        future_real = infer_future.denormalize_points(future_norm, cfg)
        future_rad = torch.tensor(future_real[:, :2], dtype=torch.float32) * torch.pi / 180.0
        d_from_prev = haversine(prev_rad.unsqueeze(0), future_rad.unsqueeze(0)).squeeze(0).cpu().numpy()
        d_steps = (haversine(future_rad[:-1].unsqueeze(0), future_rad[1:].unsqueeze(0)).squeeze(0).cpu().numpy() if future_rad.shape[0] > 1 else np.array([]))
        first_dist = float(d_from_prev[0])
        max_jump = float(np.max(d_steps)) if d_steps.size > 0 else 0.0
        score = first_dist + max_jump
        if first_dist <= reject_first_km and max_jump <= reject_step_km:
            return future_real, first_dist, float(np.max(d_from_prev)), max_jump, float(np.mean(d_steps)) if d_steps.size > 0 else 0.0, attempt
        if score < best_score:
            best_score = score
            best = (future_real, first_dist, float(np.max(d_from_prev)), max_jump, float(np.mean(d_steps)) if d_steps.size > 0 else 0.0, attempt)
    return best


def evaluate_track(track, cfg, model, args, track_index):
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

    issues = []
    result_rows = []

    with torch.no_grad():
        for sample_idx in range(args.n_samples):
            sampled = sample_safe_future(
                model,
                seqs,
                token_types,
                valid_mask,
                port_features,
                land_features,
                cfg,
                args.forecast_len,
                args.temperature,
                args.top_k,
                args.max_sample_attempts,
                args.reject_first_km,
                args.reject_step_km,
            )
            future_real, first_from_prev, max_from_prev, max_jump, mean_jump, attempts = sampled

            result_rows.append({
                'track_idx': track_index,
                'sample_idx': sample_idx,
                'first_from_prev_km': first_from_prev,
                'max_from_prev_km': max_from_prev,
                'max_step_km': max_jump,
                'mean_step_km': mean_jump,
                'attempts': attempts,
            })

            if first_from_prev > args.far_threshold:
                issues.append(f'Track {track_index} sample {sample_idx}: first predicted point {first_from_prev:.2f} km from last observed point')
            if max_from_prev > args.far_threshold * 2:
                issues.append(f'Track {track_index} sample {sample_idx}: max predicted distance {max_from_prev:.2f} km from last observed point')
            if max_jump > args.jump_threshold:
                issues.append(f'Track {track_index} sample {sample_idx}: abrupt jump {max_jump:.2f} km between forecast steps')

    return result_rows, issues


def main():
    args = parse_args()
    cfg = infer_future.Config()
    test_path = ROOT / 'data' / 'ct_dma' / 'ct_dma_test.pkl'
    with test_path.open('rb') as f:
        data = pickle.load(f)

    n_tracks = min(args.n_tracks, len(data))
    if n_tracks == 0:
        raise ValueError('No test tracks available.')

    checkpoint = Path(cfg.ckpt_path)
    model = models.TrAISformerInterpolation(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    all_rows = []
    all_issues = []
    for track_idx in range(n_tracks):
        rows, issues = evaluate_track(data[track_idx], cfg, model, args, track_idx)
        all_rows.extend(rows)
        all_issues.extend(issues)

    print('Divergence results:')
    print('track_idx,sample_idx,first_from_prev_km,max_from_prev_km,max_step_km,mean_step_km')
    for row in all_rows:
        print('{track_idx},{sample_idx},{first_from_prev_km:.3f},{max_from_prev_km:.3f},{max_step_km:.3f},{mean_step_km:.3f}'.format(**row))

    print('\nIssue summary:')
    if not all_issues:
        print('No major forecast divergence issues found for the sampled tracks.')
    else:
        for issue in all_issues:
            print('-', issue)
        print('\nRecommendation: review the model training data and conditions for large deviations, then retrain with more constrained sampling or improved context features.')


if __name__ == '__main__':
    main()
