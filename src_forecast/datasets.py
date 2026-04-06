# coding=utf-8
"""Datasets for past-only trajectory forecasting."""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

import src_interpolation.land_context as land_context
import src_interpolation.port_context as port_context
import src_interpolation.position_utils as position_utils

logger = logging.getLogger(__name__)


def log_forecast_sampling_stats(dataset, phase_name, n_samples=8192, seed=0):
    n = min(n_samples, len(dataset))
    if n <= 0:
        logger.info("Forecast sampling stats [%s]: empty dataset", phase_name)
        return
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=(len(dataset) < n_samples))
    pasts, futures = [], []
    for i in idxs:
        _, _, _, _, _, pl, fl, _, _, _, _, _ = dataset[int(i)]
        pasts.append(int(pl))
        futures.append(int(fl))
    pasts = np.array(pasts, dtype=np.int64)
    futures = np.array(futures, dtype=np.int64)

    def _hist(name, arr):
        lo, hi = int(arr.min()), int(arr.max())
        counts = np.bincount(arr, minlength=hi + 1)
        nz = np.nonzero(counts)[0]
        parts = [f"{int(k)}:{int(counts[k])}" for k in nz]
        return f"{name} min={lo} max={hi} mean={arr.mean():.2f} | " + " ".join(parts)

    logger.info(
        "Forecast sampling stats [%s] (n=%d): %s; %s",
        phase_name,
        n,
        _hist("past", pasts),
        _hist("future", futures),
    )


class AISForecastDataset(Dataset):
    """Sample windows for past-only forecasting."""

    def __init__(
        self,
        l_data,
        max_seqlen=120,
        min_past_points=1,
        max_past_points=40,
        min_future_points=1,
        max_future_points=40,
        edge_case_prob=0.2,
        samples_per_track=1,
        seed=42,
        config=None,
    ):
        self.max_seqlen = max_seqlen
        self.min_past_points = min_past_points
        self.max_past_points = max_past_points
        self.min_future_points = min_future_points
        self.max_future_points = max_future_points
        self.edge_case_prob = edge_case_prob
        self.samples_per_track = samples_per_track
        self.seed = seed
        self.config = config
        self.port_encoder = None
        self.port_context_size = 0
        self.land_encoder = None
        self.land_context_size = 0
        if config is not None and getattr(config, "use_port_context", False):
            self.port_encoder = port_context.PortContextEncoder.from_config(config)
            self.port_context_size = self.port_encoder.context_size
        if config is not None and getattr(config, "use_land_context", False):
            self.land_encoder = land_context.LandContextEncoder.from_config(config)
            self.land_context_size = self.land_encoder.context_size

        self.min_total_points = self.min_past_points + self.min_future_points
        self.l_data = [
            vessel for vessel in l_data if len(vessel["traj"]) >= self.min_total_points
        ]
        self.sample_refs = [
            (track_idx, sample_idx)
            for track_idx in range(len(self.l_data))
            for sample_idx in range(self.samples_per_track)
        ]

        if config is not None and (self.port_encoder is not None or self.land_encoder is not None):
            logger.info("Pre-computing port/land context for %d tracks...", len(self.l_data))
            for vessel in self.l_data:
                traj = np.clip(vessel["traj"][:, :4], 0.0, 0.9999)
                real_lats, real_lons = position_utils.source_positions_to_real_np(
                    traj[:, 0], traj[:, 1], config
                )
                if self.port_encoder is not None:
                    vessel["_port_features"] = self.port_encoder.encode_positions(real_lats, real_lons)
                if self.land_encoder is not None:
                    vessel["_land_features"] = self.land_encoder.encode_positions(real_lats, real_lons)
            logger.info("Pre-computation complete.")

    def __len__(self):
        return len(self.sample_refs)

    def _make_rng(self, idx):
        track_idx, sample_idx = self.sample_refs[idx]
        return np.random.default_rng(self.seed + 1000003 * track_idx + sample_idx)

    def _sample_lengths(self, track_len, rng):
        max_window = min(track_len, self.max_seqlen)
        use_edge_case = (
            rng.random() < self.edge_case_prob
            and max_window >= self.min_past_points + self.min_future_points + 2
        )

        if use_edge_case:
            past_len = self.min_past_points
            future_len = self.min_future_points
            return past_len, future_len

        future_upper = min(self.max_future_points, max_window - self.min_past_points)
        future_len = int(rng.integers(self.min_future_points, future_upper + 1))

        past_upper = min(self.max_past_points, max_window - future_len)
        past_len = int(rng.integers(self.min_past_points, past_upper + 1))

        return past_len, future_len

    def __getitem__(self, idx):
        track_idx, _ = self.sample_refs[idx]
        rng = self._make_rng(idx)
        vessel = self.l_data[track_idx]
        traj = vessel["traj"].copy()
        traj[:, :4] = np.clip(traj[:, :4], 0.0, 0.9999)

        track_len = len(traj)
        past_len, future_len = self._sample_lengths(track_len, rng)
        total_len = past_len + future_len
        start_idx = int(rng.integers(0, track_len - total_len + 1))
        window = traj[start_idx : start_idx + total_len]

        if self.config is not None and position_utils.uses_local_position_frame(self.config):
            real_lats, real_lons = position_utils.source_positions_to_real_np(
                window[:, 0], window[:, 1], self.config
            )
            origin_lat = float(real_lats[past_len - 1])
            origin_lon = float(real_lons[past_len - 1])
            lat_norm, lon_norm, _, _ = position_utils.real_positions_to_model_norm_np(
                real_lats,
                real_lons,
                self.config,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
            )
            window[:, 0] = lat_norm
            window[:, 1] = lon_norm

        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        seq[:total_len] = window[:, :4]

        token_types = np.zeros(self.max_seqlen, dtype=np.int64)
        token_types[:past_len] = 1
        token_types[past_len:total_len] = 2

        valid_mask = np.zeros(self.max_seqlen, dtype=np.float32)
        valid_mask[:total_len] = 1.0

        target_mask = np.zeros(self.max_seqlen, dtype=np.float32)
        target_mask[past_len:total_len] = 1.0

        time_seq = np.zeros(self.max_seqlen, dtype=np.int64)
        if window.shape[1] > 4:
            time_seq[:total_len] = window[:, 4].astype(np.int64)

        port_features = np.zeros((self.max_seqlen, self.port_context_size), dtype=np.float32)
        land_features = np.zeros((self.max_seqlen, self.land_context_size), dtype=np.float32)
        if total_len > 0:
            future_mask = token_types[:total_len] == 2
            if self.port_encoder is not None and "_port_features" in vessel:
                sliced = vessel["_port_features"][start_idx : start_idx + total_len].copy()
                sliced[future_mask] = 0.0
                port_features[:total_len] = sliced
            if self.land_encoder is not None and "_land_features" in vessel:
                sliced = vessel["_land_features"][start_idx : start_idx + total_len].copy()
                sliced[future_mask] = 0.0
                land_features[:total_len] = sliced

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(token_types, dtype=torch.long),
            torch.tensor(valid_mask, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
            torch.tensor(total_len, dtype=torch.long),
            torch.tensor(past_len, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
            torch.tensor(future_len, dtype=torch.long),
            torch.tensor(vessel["mmsi"], dtype=torch.long),
            torch.tensor(time_seq, dtype=torch.long),
            torch.tensor(port_features, dtype=torch.float32),
            torch.tensor(land_features, dtype=torch.float32),
        )


def build_forecast_sequence(
    prev_seq,
    future_len,
    max_seqlen,
    port_encoder=None,
    land_encoder=None,
    prev_real_points=None,
    port_context_size=None,
    land_context_size=None,
    config=None,
):
    prev_seq = np.asarray(prev_seq, dtype=np.float32)

    total_len = len(prev_seq) + future_len
    if total_len > max_seqlen:
        raise ValueError(
            f"Forecast window ({total_len}) exceeds max_seqlen ({max_seqlen})."
        )

    seq = np.zeros((max_seqlen, 4), dtype=np.float32)
    if config is not None and position_utils.uses_local_position_frame(config):
        if prev_real_points is None:
            raise ValueError("Real previous positions are required for local-frame forecast sequence building.")
        prev_real_points = np.asarray(prev_real_points, dtype=np.float32)
        lat_norm, lon_norm, _, _ = position_utils.real_positions_to_model_norm_np(
            prev_real_points[:, 0],
            prev_real_points[:, 1],
            config,
        )
        seq[:len(prev_seq), 0] = lat_norm
        seq[:len(prev_seq), 1] = lon_norm
        seq[:len(prev_seq), 2:] = np.clip(prev_seq[:, 2:], 0.0, 0.9999)
    else:
        seq[:len(prev_seq)] = np.clip(prev_seq, 0.0, 0.9999)

    token_types = np.zeros(max_seqlen, dtype=np.int64)
    token_types[:len(prev_seq)] = 1
    token_types[len(prev_seq) : total_len] = 2

    valid_mask = np.zeros(max_seqlen, dtype=np.float32)
    valid_mask[:total_len] = 1.0

    target_mask = np.zeros(max_seqlen, dtype=np.float32)
    target_mask[len(prev_seq) : total_len] = 1.0

    if port_context_size is None:
        port_context_size = 0 if port_encoder is None else int(port_encoder.context_size)
    port_features = np.zeros((max_seqlen, int(port_context_size)), dtype=np.float32)
    if land_context_size is None:
        land_context_size = 0 if land_encoder is None else int(land_encoder.context_size)
    land_features = np.zeros((max_seqlen, int(land_context_size)), dtype=np.float32)

    if port_encoder is not None and int(port_context_size) > 0:
        if prev_real_points is None:
            raise ValueError("Real previous positions are required to build port context.")
        prev_real_points = np.asarray(prev_real_points, dtype=np.float32)
        port_features[: len(prev_seq)] = port_encoder.encode_positions(
            prev_real_points[:, 0],
            prev_real_points[:, 1],
            token_types=np.full(len(prev_seq), 1, dtype=np.int64),
        )

    if land_encoder is not None and int(land_context_size) > 0:
        if prev_real_points is None:
            raise ValueError("Real previous positions are required to build land context.")
        prev_real_points = np.asarray(prev_real_points, dtype=np.float32)
        land_features[: len(prev_seq)] = land_encoder.encode_positions(
            prev_real_points[:, 0],
            prev_real_points[:, 1],
            token_types=np.full(len(prev_seq), 1, dtype=np.int64),
        )

    return (
        torch.tensor(seq, dtype=torch.float32).unsqueeze(0),
        torch.tensor(token_types, dtype=torch.long).unsqueeze(0),
        torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0),
        torch.tensor(target_mask, dtype=torch.float32).unsqueeze(0),
        torch.tensor(port_features, dtype=torch.float32).unsqueeze(0),
        torch.tensor(land_features, dtype=torch.float32).unsqueeze(0),
    )
