#!/usr/bin/env python
# coding=utf-8

"""Evaluate the saved forecast model on the held-out test set."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src_forecast.trAISformer import Config, load_phase_data
import src_forecast.datasets as fds
from src_interpolation import models, trainers
import utils


def main():
    cfg = Config()
    test_data = load_phase_data(cfg.testset_name)
    print(f"Loaded test tracks: {len(test_data)}")

    test_ds = fds.AISForecastDataset(
        test_data,
        max_seqlen=cfg.max_seqlen,
        min_past_points=cfg.min_past_points,
        max_past_points=cfg.max_past_points,
        min_future_points=cfg.min_future_points,
        max_future_points=cfg.max_future_points,
        edge_case_prob=cfg.edge_case_prob,
        samples_per_track=cfg.eval_samples_per_track,
        seed=cfg.seed + 20000,
        config=cfg,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = models.TrAISformerInterpolation(cfg)
    checkpoint = Path(cfg.ckpt_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Forecast checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    model = model.to(cfg.device)
    model.eval()

    total_err = 0.0
    total_count = 0
    step_err = np.zeros(cfg.max_future_points, dtype=np.float64)
    step_count = np.zeros(cfg.max_future_points, dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating", unit="batch"):
            (
                seqs,
                token_types,
                valid_masks,
                target_masks,
                seqlens,
                past_lens,
                gap_lens,
                future_lens,
                mmsis,
                time_seqs,
                port_features,
                land_features,
            ) = batch
            seqs = seqs.to(cfg.device)
            token_types = token_types.to(cfg.device)
            valid_masks = valid_masks.to(cfg.device)
            target_masks = target_masks.to(cfg.device)
            port_features = port_features.to(cfg.device)
            land_features = land_features.to(cfg.device)

            preds = trainers.predict_gap(
                model,
                seqs,
                token_types,
                valid_masks,
                port_context=port_features,
                land_context=land_features,
                sample=False,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )

            input_coords = seqs[:, :, :2].clone()
            input_coords[..., 0] = input_coords[..., 0] * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
            input_coords[..., 1] = input_coords[..., 1] * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
            input_coords = input_coords * torch.pi / 180.0

            pred_coords = preds[:, :, :2].clone()
            pred_coords[..., 0] = pred_coords[..., 0] * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
            pred_coords[..., 1] = pred_coords[..., 1] * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
            pred_coords = pred_coords * torch.pi / 180.0

            d = utils.haversine(input_coords, pred_coords)
            total_err += (d * target_masks).sum().item()
            total_count += target_masks.sum().item()

            d_cpu = d.detach().cpu().numpy()
            past_cpu = past_lens.numpy()
            future_cpu = future_lens.numpy()
            for idx in range(d_cpu.shape[0]):
                start = int(past_cpu[idx])
                future_len = int(future_cpu[idx])
                errors = d_cpu[idx, start : start + future_len]
                step_err[:future_len] += errors
                step_count[:future_len] += 1

    overall = total_err / max(total_count, 1)
    print(f"Overall mean forecast error (km): {overall:.4f}")
    print("Step 1-10 mean errors (km):")
    for step in range(10):
        if step_count[step] > 0:
            print(f"  step {step+1}: {step_err[step] / step_count[step]:.4f} ({step_count[step]} points)")
        else:
            print(f"  step {step+1}: no samples")


if __name__ == '__main__':
    main()
