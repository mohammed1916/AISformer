#!/usr/bin/env python
# coding: utf-8

"""Train and evaluate the interpolation variant of TrAISformer."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import trainers
import utils
from config_trAISformer import Config


cf = Config()
utils.set_seed(cf.seed)
torch.pi = torch.acos(torch.zeros(1)).item() * 2


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
        )
        aisdls[phase] = DataLoader(
            aisdatasets[phase],
            batch_size=cf.batch_size,
            shuffle=(phase == "train"),
            num_workers=cf.num_workers,
        )

    approx_gap = 0.5 * (cf.min_gap_points + cf.max_gap_points)
    cf.final_tokens = max(1, int(len(aisdatasets["train"]) * approx_gap * cf.max_epochs))
    return data, aisdatasets, aisdls


def evaluate(model, aisdls):
    v_ranges = torch.tensor(
        [model.lat_max - model.lat_min, model.lon_max - model.lon_min, model.sog_range, 360.0],
        device=cf.device,
    )
    v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0.0, 0.0], device=cf.device)

    gap_error_sum = np.zeros(cf.max_gap_points, dtype=np.float64)
    gap_error_count = np.zeros(cf.max_gap_points, dtype=np.float64)
    global_error_sum = 0.0
    global_gap_count = 0.0

    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
        for _, batch in pbar:
            seqs, token_types, valid_masks, target_masks, seqlens, past_lens, gap_lens, future_lens, mmsis, time_seqs = batch
            seqs = seqs.to(cf.device)
            token_types = token_types.to(cf.device)
            valid_masks = valid_masks.to(cf.device)
            target_masks = target_masks.to(cf.device)

            preds = trainers.predict_gap(
                model,
                seqs,
                token_types,
                valid_masks,
                sample=False,
                temperature=cf.temperature,
                top_k=cf.top_k,
            )

            input_coords = (seqs * v_ranges + v_roi_min) * torch.pi / 180
            pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
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
    print("\n== TrAISformer interpolation training ==")
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print("======= Create directory to store trained models: " + cf.savedir)
    else:
        print("======= Directory to store trained models: " + cf.savedir)
    utils.new_log(cf.savedir, "log")

    data, aisdatasets, aisdls = build_datasets()

    model = models.TrAISformerInterpolation(cf)
    model.lat_min = cf.lat_min
    model.lat_max = cf.lat_max
    model.lon_min = cf.lon_min
    model.lon_max = cf.lon_max
    model.sog_range = cf.sog_range

    trainer = trainers.Trainer(
        model,
        aisdatasets["train"],
        aisdatasets["valid"],
        cf,
        savedir=cf.savedir,
        device=cf.device,
        aisdls=aisdls,
    )

    if cf.retrain or not os.path.exists(cf.ckpt_path):
        trainer.train()

    model.load_state_dict(torch.load(cf.ckpt_path, map_location=cf.device))
    model = model.to(cf.device)
    evaluate(model, aisdls)
