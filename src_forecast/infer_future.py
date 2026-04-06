#!/usr/bin/env python
# coding=utf-8

"""CLI for forecasting future trajectory points from past AIS observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src_forecast import datasets
from src_interpolation import models, trainers
import src_interpolation.land_context as land_context
import src_interpolation.port_context as port_context
from src_forecast.config_trAISformer import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a future trajectory given a history of AIS points.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained forecast checkpoint.")
    parser.add_argument(
        "--prev-json",
        type=str,
        default=None,
        help="JSON array for the previous trajectory.",
    )
    parser.add_argument("--prev-file", type=str, default=None, help="JSON file containing the previous trajectory.")
    parser.add_argument("--future-len", type=int, required=True, help="Number of future points to predict.")
    parser.add_argument(
        "--input-space",
        choices=("normalized", "real"),
        default="normalized",
        help="Whether the provided points are normalized [0,1) values or real-world values.",
    )
    parser.add_argument("--sample", action="store_true", help="Sample from the output distribution instead of argmax.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of probable trajectories to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature. Lower values reduce unlikely jumps.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k filtering for decoding. Smaller values improve stability.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional file path to write JSON output.")
    return parser.parse_args()


def load_json_points(json_text=None, file_path=None, label="trajectory"):
    if json_text is not None:
        payload = json.loads(json_text)
    elif file_path is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raise ValueError(f"Provide either inline JSON or a JSON file for {label}.")

    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON list.")

    points = []
    for idx, item in enumerate(payload):
        if isinstance(item, dict):
            point = [item[key] for key in ("lat", "lon", "sog", "cog")]
        else:
            point = item

        if not isinstance(point, list) or len(point) != 4:
            raise ValueError(f"{label}[{idx}] must contain exactly 4 values: [lat, lon, sog, cog].")

        points.append([float(value) for value in point])

    return np.asarray(points, dtype=np.float32)


def to_real_points(points, config, input_space):
    if input_space == "real":
        return np.asarray(points, dtype=np.float32)

    normalized = np.clip(np.asarray(points, dtype=np.float32), 0.0, 0.9999)
    real = np.empty_like(normalized, dtype=np.float32)
    real[:, 0] = normalized[:, 0] * (config.lat_max - config.lat_min) + config.lat_min
    real[:, 1] = normalized[:, 1] * (config.lon_max - config.lon_min) + config.lon_min
    real[:, 2] = normalized[:, 2] * config.sog_range
    real[:, 3] = normalized[:, 3] * 360.0
    return real


def normalize_points(points, config):
    real = np.asarray(points, dtype=np.float32)
    normalized = np.empty_like(real, dtype=np.float32)
    normalized[:, 0] = (real[:, 0] - config.lat_min) / (config.lat_max - config.lat_min)
    normalized[:, 1] = (real[:, 1] - config.lon_min) / (config.lon_max - config.lon_min)
    normalized[:, 2] = np.clip(real[:, 2] / config.sog_range, 0.0, 0.9999)
    normalized[:, 3] = np.clip((real[:, 3] % 360.0) / 360.0, 0.0, 0.9999)
    return normalized


def denormalize_points(points, config):
    real = np.empty_like(points, dtype=np.float32)
    real[:, 0] = points[:, 0] * (config.lat_max - config.lat_min) + config.lat_min
    real[:, 1] = points[:, 1] * (config.lon_max - config.lon_min) + config.lon_min
    real[:, 2] = points[:, 2] * config.sog_range
    real[:, 3] = points[:, 3] * 360.0
    return real


def find_checkpoint(config, explicit_path=None):
    if explicit_path:
        checkpoint = Path(explicit_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    default_ckpt = Path(config.ckpt_path)
    if default_ckpt.is_file():
        return default_ckpt

    candidates = sorted(
        Path(".").glob("results_forecast/**/*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No forecast checkpoint found. Train the model first or pass --checkpoint."
    )


def main():
    args = parse_args()
    config = Config()

    prev_points = load_json_points(args.prev_json, args.prev_file, label="previous trajectory")
    if len(prev_points) == 0:
        raise ValueError("Previous trajectory must contain at least one point.")
    if args.future_len < 1:
        raise ValueError("--future-len must be at least 1.")

    prev_real = to_real_points(prev_points, config, args.input_space)
    prev_norm = normalize_points(prev_real, config)

    port_encoder = None
    land_encoder = None
    if getattr(config, "use_port_context", False):
        port_encoder = port_context.PortContextEncoder.from_config(config)
    if getattr(config, "use_land_context", False):
        land_encoder = land_context.LandContextEncoder.from_config(config)

    seqs, token_types, valid_mask, target_mask, port_features, land_features = datasets.build_forecast_sequence(
        prev_norm,
        args.future_len,
        config.max_seqlen,
        port_encoder=port_encoder,
        land_encoder=land_encoder,
        prev_real_points=prev_real[:, :2],
        port_context_size=getattr(config, "port_context_size", 0),
        land_context_size=getattr(config, "land_context_size", 0),
    )

    checkpoint = find_checkpoint(config, args.checkpoint)
    model = models.TrAISformerInterpolation(config)
    model.load_state_dict(torch.load(checkpoint, map_location=config.device))
    model = model.to(config.device)

    seqs = seqs.to(config.device)
    token_types = token_types.to(config.device)
    valid_mask = valid_mask.to(config.device)
    port_features = port_features.to(config.device)
    land_features = land_features.to(config.device)

    if args.n_samples < 1:
        raise ValueError("--n-samples must be at least 1.")
    if args.n_samples > 1:
        args.sample = True

    all_predicted_norm = []
    all_predicted_real = []
    for sample_idx in range(args.n_samples):
        completed = trainers.predict_gap(
            model,
            seqs,
            token_types,
            valid_mask,
            port_context=port_features,
            land_context=land_features,
            sample=args.sample,
            temperature=args.temperature if args.temperature is not None else config.temperature,
            top_k=args.top_k if args.top_k is not None else config.top_k,
        )
        predicted_future_norm = completed[0, len(prev_norm) : len(prev_norm) + args.future_len].detach().cpu().numpy()
        predicted_future_real = denormalize_points(predicted_future_norm, config)
        all_predicted_norm.append(predicted_future_norm.tolist())
        all_predicted_real.append(predicted_future_real.tolist())

    result = {
        "checkpoint": str(checkpoint),
        "future_len": args.future_len,
        "input_space": args.input_space,
        "predicted_futures_normalized": all_predicted_norm,
        "predicted_futures_real": all_predicted_real,
    }
    if args.n_samples == 1:
        result["predicted_future_normalized"] = all_predicted_norm[0]
        result["predicted_future_real"] = all_predicted_real[0]

    output_text = json.dumps(result, indent=2)
    print(output_text)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
