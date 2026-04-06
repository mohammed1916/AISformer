#!/usr/bin/env python
# coding=utf-8

"""Generate a PNG showing multiple sampled forecast trajectories."""

from pathlib import Path
import sys

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


def main():
    cfg = infer_future.Config()
    test_path = ROOT / 'data' / 'ct_dma' / 'ct_dma_test.pkl'
    with test_path.open('rb') as f:
        data = pickle.load(f)

    prev_points = np.asarray(data[0]['traj'][:10, :4], dtype=np.float32)
    prev_real = infer_future.to_real_points(prev_points, cfg, 'real')
    prev_norm = infer_future.normalize_points(prev_real, cfg)

    port_encoder = None
    land_encoder = None
    if getattr(cfg, 'use_port_context', False):
        port_encoder = port_context.PortContextEncoder.from_config(cfg)
    if getattr(cfg, 'use_land_context', False):
        land_encoder = land_context.LandContextEncoder.from_config(cfg)

    seqs, token_types, valid_mask, target_mask, port_features, land_features = infer_future.datasets.build_forecast_sequence(
        prev_norm,
        10,
        cfg.max_seqlen,
        port_encoder=port_encoder,
        land_encoder=land_encoder,
        prev_real_points=prev_real[:, :2],
        port_context_size=getattr(cfg, 'port_context_size', 0),
        land_context_size=getattr(cfg, 'land_context_size', 0),
    )

    checkpoint = Path(cfg.ckpt_path)
    model = models.TrAISformerInterpolation(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    seqs = seqs.to(cfg.device)
    token_types = token_types.to(cfg.device)
    valid_mask = valid_mask.to(cfg.device)
    port_features = port_features.to(cfg.device)
    land_features = land_features.to(cfg.device)

    samples = []
    n_samples = 3
    for _ in range(n_samples):
        completed = trainers.predict_gap(
            model,
            seqs,
            token_types,
            valid_mask,
            port_context=port_features,
            land_context=land_features,
            sample=True,
            temperature=1.0,
            top_k=20,
        )
        future_norm = completed[0, len(prev_norm) : len(prev_norm) + 10].detach().cpu().numpy()
        samples.append(infer_future.denormalize_points(future_norm, cfg))

    plt.figure(figsize=(6, 6), dpi=150)
    plt.plot(prev_real[:, 1], prev_real[:, 0], '-k', marker='x', label='history')
    for idx, future_real in enumerate(samples):
        plt.plot(future_real[:, 1], future_real[:, 0], '-o', label=f'sample {idx+1}')

    plt.legend(loc='best')
    plt.title('Multiple Forecast Samples')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    out_path = ROOT / 'results_forecast' / 'forecast_sample_paths.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f'Saved forecast sample PNG: {out_path}')


if __name__ == '__main__':
    main()
