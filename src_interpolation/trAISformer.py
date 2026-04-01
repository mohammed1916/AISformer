#!/usr/bin/env python
# coding: utf-8

"""Train and evaluate the interpolation variant of TrAISformer."""

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import position_utils
import trainers
import utils
from config_trAISformer import Config


cf = Config()
utils.set_seed(cf.seed)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# ── Ada Lovelace / CUDA global optimizations ──────────────────────────────────
# TF32 uses 10-bit mantissa on Tensor Cores: ~3× faster matmuls, negligible loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Let cuDNN auto-select fastest conv kernels for fixed input shapes
torch.backends.cudnn.benchmark = True


def load_phase_data(filename):
    datapath = os.path.join(cf.datadir, filename)
    print(f"Loading {datapath}...")
    with open(datapath, "rb") as f:
        l_pred_errors = pickle.load(f)

    moving_threshold = 0.05
    for vessel in l_pred_errors:
        try:
            moving_idx = np.where(vessel["traj"][:, 2] > moving_threshold)[0][0]
        except Exception:
            moving_idx = len(vessel["traj"]) - 1
        vessel["traj"] = vessel["traj"][moving_idx:, :]

    min_required = max(
        cf.min_seqlen,
        cf.min_past_points + cf.min_gap_points + cf.min_future_points,
    )
    phase_data = [
        vessel
        for vessel in l_pred_errors
        if not np.isnan(vessel["traj"]).any() and len(vessel["traj"]) >= min_required
    ]
    print(len(l_pred_errors), len(phase_data))
    print(f"Length: {len(phase_data)}")
    return phase_data


def build_datasets():
    data = {}
    aisdatasets = {}
    aisdls = {}
    phase_filenames = {
        "train": cf.trainset_name,
        "valid": cf.validset_name,
        "test": cf.testset_name,
    }
    samples_per_track = {
        "train": cf.train_samples_per_track,
        "valid": cf.eval_samples_per_track,
        "test": cf.eval_samples_per_track,
    }

    for phase, filename in phase_filenames.items():
        data[phase] = load_phase_data(filename)
        aisdatasets[phase] = datasets.AISInterpolationDataset(
            data[phase],
            max_seqlen=cf.max_seqlen,
            min_past_points=cf.min_past_points,
            max_past_points=cf.max_past_points,
            min_future_points=cf.min_future_points,
            max_future_points=cf.max_future_points,
            min_gap_points=cf.min_gap_points,
            max_gap_points=cf.max_gap_points,
            edge_case_prob=cf.edge_case_prob,
            samples_per_track=samples_per_track[phase],
            seed=cf.seed + {"train": 0, "valid": 10000, "test": 20000}[phase],
            config=cf,
        )
        aisdls[phase] = DataLoader(
            aisdatasets[phase],
            batch_size=cf.batch_size,
            shuffle=(phase == "train"),
            num_workers=cf.num_workers,
        )

        if getattr(cf, "log_gap_sampling", False):
            datasets.log_gap_sampling_stats(
                aisdatasets[phase],
                phase,
                n_samples=getattr(cf, "log_gap_sample_budget", 8192),
                seed=cf.seed + {"train": 0, "valid": 7, "test": 13}[phase],
            )

    approx_gap = 0.5 * (cf.min_gap_points + cf.max_gap_points)
    cf.final_tokens = max(1, int(len(aisdatasets["train"]) * approx_gap * cf.max_epochs))
    return data, aisdatasets, aisdls


def evaluate(model, aisdls):
    gap_error_sum = np.zeros(cf.max_gap_points, dtype=np.float64)
    gap_error_count = np.zeros(cf.max_gap_points, dtype=np.float64)
    global_error_sum = 0.0
    global_gap_count = 0.0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
        for _, batch in pbar:
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
            seqs = seqs.to(cf.device)
            token_types = token_types.to(cf.device)
            valid_masks = valid_masks.to(cf.device)
            target_masks = target_masks.to(cf.device)
            port_features = port_features.to(cf.device)
            land_features = land_features.to(cf.device)
            preds = trainers.predict_gap(
                model,
                seqs,
                token_types,
                valid_masks,
                port_context=port_features,
                land_context=land_features,
                sample=False,
                temperature=cf.temperature,
                top_k=cf.top_k,
            )

            # Global normalization: convert [0,1) to real lat/lon
            input_coords = seqs[:, :, :2].clone()
            input_coords[..., 0] = input_coords[..., 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
            input_coords[..., 1] = input_coords[..., 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
            input_coords = input_coords * torch.pi / 180.0

            pred_coords = preds[:, :, :2].clone()
            pred_coords[..., 0] = pred_coords[..., 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
            pred_coords[..., 1] = pred_coords[..., 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
            pred_coords = pred_coords * torch.pi / 180.0

            d = utils.haversine(input_coords, pred_coords)

            global_error_sum += (d * target_masks).sum().item()
            global_gap_count += target_masks.sum().item()

            d_cpu = d.detach().cpu().numpy()
            past_cpu = past_lens.numpy()
            gap_cpu = gap_lens.numpy()
            for idx in range(d_cpu.shape[0]):
                start = int(past_cpu[idx])
                gap_len = int(gap_cpu[idx])
                gap_errors = d_cpu[idx, start:start + gap_len]
                gap_error_sum[:gap_len] += gap_errors
                gap_error_count[:gap_len] += 1

    mean_gap_errors = np.divide(
        gap_error_sum,
        np.maximum(gap_error_count, 1),
        out=np.zeros_like(gap_error_sum),
        where=gap_error_count > 0,
    )
    overall_gap_error = global_error_sum / max(global_gap_count, 1.0)

    plt.figure(figsize=(9, 6), dpi=150)
    valid_steps = gap_error_count > 0
    plt.plot(np.arange(1, cf.max_gap_points + 1)[valid_steps], mean_gap_errors[valid_steps])
    plt.xlabel("Step inside missing span")
    plt.ylabel("Interpolation error (km)")
    plt.title(f"Mean gap interpolation error = {overall_gap_error:.4f} km")
    plt.xlim([1, max(2, cf.max_gap_points)])
    plt.savefig(os.path.join(cf.savedir, "gap_prediction_error.png"))
    plt.close()

    print(f"Mean gap interpolation error: {overall_gap_error:.4f} km")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrAISformer Interpolation Training")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--retrain", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.max_epochs is not None:
        cf.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cf.batch_size = args.batch_size
    if args.learning_rate is not None:
        cf.learning_rate = args.learning_rate
    if args.warmup_tokens is not None:
        cf.warmup_tokens = args.warmup_tokens
    if args.n_samples is not None:
        cf.n_samples = args.n_samples
    if args.retrain is not None:
        cf.retrain = args.retrain

    print("\n== TrAISformer interpolation training ==")
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print("======= Create directory to store trained models: " + cf.savedir)
    else:
        print("======= Directory to store trained models: " + cf.savedir)
    utils.new_log(cf.savedir, "log")

    logging.info(
        "Interpolation config: gap [%d,%d], past [%d,%d], future [%d,%d], edge_case_prob=%s, "
        "bins lat/lon/sog/cog=%d/%d/%d/%d, port_context=%s(%d), land_context=%s(%d), lr_decay=%s weight_decay=%s",
        cf.min_gap_points,
        cf.max_gap_points,
        cf.min_past_points,
        cf.max_past_points,
        cf.min_future_points,
        cf.max_future_points,
        cf.edge_case_prob,
        cf.lat_size,
        cf.lon_size,
        cf.sog_size,
        cf.cog_size,
        getattr(cf, "use_port_context", False),
        getattr(cf, "port_context_size", 0),
        getattr(cf, "use_land_context", False),
        getattr(cf, "land_context_size", 0),
        cf.lr_decay,
        cf.weight_decay,
    )

    data, aisdatasets, aisdls = build_datasets()

    model = models.TrAISformerInterpolation(cf)

    # torch.compile: fuses ops into Triton/CUDA kernels (sm_89 Ada Lovelace)
    # reduce-overhead uses CUDA graphs to eliminate per-iter Python launch overhead
    if getattr(cf, "use_compile", True) and torch.cuda.is_available():
        logging.info("Compiling model with torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    trainer = trainers.Trainer(
        model,
        aisdatasets["train"],
        aisdatasets["valid"],
        cf,
        savedir=cf.savedir,
        device=cf.device,
        aisdls=aisdls,
    )

    if args.eval_only:
        model.load_state_dict(torch.load(cf.ckpt_path, map_location=cf.device))
        model = model.to(cf.device)
        evaluate(model, aisdls)
    else:
        if cf.retrain or not os.path.exists(cf.ckpt_path):
            trainer.train()

        model.load_state_dict(torch.load(cf.ckpt_path, map_location=cf.device))
        model = model.to(cf.device)
        evaluate(model, aisdls)
