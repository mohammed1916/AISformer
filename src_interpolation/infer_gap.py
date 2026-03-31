#!/usr/bin/env python
# coding: utf-8

"""CLI for gap interpolation from previous and future trajectory segments."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import datasets
import land_context
import models
import port_context
import position_utils
import trainers
from config_trAISformer import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the missing middle trajectory between a previous and future segment."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained interpolation checkpoint.")
    parser.add_argument("--prev-json", type=str, default=None, help="JSON array for the previous trajectory.")
    parser.add_argument("--next-json", type=str, default=None, help="JSON array for the future trajectory.")
    parser.add_argument("--prev-file", type=str, default=None, help="JSON file containing the previous trajectory.")
    parser.add_argument("--next-file", type=str, default=None, help="JSON file containing the future trajectory.")
    parser.add_argument("--gap-len", type=int, required=True, help="Number of missing points to predict.")
    parser.add_argument(
        "--input-space",
        choices=("normalized", "real"),
        default="normalized",
        help="Whether the provided points are normalized [0,1) values or real-world values.",
    )
    parser.add_argument("--sample", action="store_true", help="Sample from the output distribution instead of argmax.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Defaults to config.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k filtering for decoding.")
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


def normalize_points(points, config, origin_lat=None, origin_lon=None):
    real = np.asarray(points, dtype=np.float32)
    normalized = np.empty_like(real, dtype=np.float32)
    normalized[:, 0] = (real[:, 0] - config.lat_min) / (config.lat_max - config.lat_min)
    normalized[:, 1] = (real[:, 1] - config.lon_min) / (config.lon_max - config.lon_min)
    normalized[:, 2] = np.clip(real[:, 2] / config.sog_range, 0.0, 0.9999)
    normalized[:, 3] = np.clip((real[:, 3] % 360.0) / 360.0, 0.0, 0.9999)
    return normalized


def denormalize_points(points, config, origin_lat=None, origin_lon=None):
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
        Path(".").glob("results_interpolation/**/*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No interpolation checkpoint found. Train the model first or pass --checkpoint."
    )


def main():
    args = parse_args()
    config = Config()

    prev_points = load_json_points(args.prev_json, args.prev_file, label="previous trajectory")
    next_points = load_json_points(args.next_json, args.next_file, label="future trajectory")
    if len(prev_points) == 0 or len(next_points) == 0:
        raise ValueError("Both previous and future trajectories must contain at least one point.")
    if args.gap_len < 1:
        raise ValueError("--gap-len must be at least 1.")

    prev_real = to_real_points(prev_points, config, args.input_space)
    next_real = to_real_points(next_points, config, args.input_space)

    prev_norm = normalize_points(prev_real, config)
    next_norm = normalize_points(next_real, config)

    encoder = None
    land_encoder = None
    if getattr(config, "use_port_context", False):
        encoder = port_context.PortContextEncoder.from_config(config)
    if getattr(config, "use_land_context", False):
        land_encoder = land_context.LandContextEncoder.from_config(config)

    seqs, token_types, valid_mask, target_mask, port_features, land_features = datasets.build_interpolation_sequence(
        prev_norm,
        next_norm,
        args.gap_len,
        config.max_seqlen,
        port_encoder=encoder,
        land_encoder=land_encoder,
        prev_real_points=prev_real[:, :2],
        next_real_points=next_real[:, :2],
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

    gap_start = len(prev_norm)
    gap_end = gap_start + args.gap_len
    predicted_gap_norm = completed[0, gap_start:gap_end].detach().cpu().numpy()
    predicted_gap_real = denormalize_points(
        predicted_gap_norm,
        config,
    )

    result = {
        "checkpoint": str(checkpoint),
        "gap_len": args.gap_len,
        "input_space": args.input_space,
        "origin": None,
        "predicted_gap_normalized": predicted_gap_norm.tolist(),
        "predicted_gap_real": predicted_gap_real.tolist(),
    }

    output_text = json.dumps(result, indent=2)
    print(output_text)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
