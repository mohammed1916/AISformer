# coding=utf-8
"""Datasets for trajectory interpolation."""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

import position_utils

logger = logging.getLogger(__name__)


def log_gap_sampling_stats(dataset, phase_name, n_samples=8192, seed=0):
    """Log empirical past / gap / future length distribution from the dataset sampler."""
    n = min(n_samples, len(dataset))
    if n <= 0:
        logger.info("gap stats [%s]: empty dataset", phase_name)
        return
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=(len(dataset) < n_samples))
    pasts, gaps, futures = [], [], []
    for i in idxs:
        _, _, _, _, _, pl, gl, fl, _, _, _, _ = dataset[int(i)]
        pasts.append(int(pl))
        gaps.append(int(gl))
        futures.append(int(fl))
    pasts = np.array(pasts, dtype=np.int64)
    gaps = np.array(gaps, dtype=np.int64)
    futures = np.array(futures, dtype=np.int64)

    def _hist(name, arr):
        lo, hi = int(arr.min()), int(arr.max())
        counts = np.bincount(arr, minlength=hi + 1)
        nz = np.nonzero(counts)[0]
        parts = [f"{int(k)}:{int(counts[k])}" for k in nz]
        return f"{name} min={lo} max={hi} mean={arr.mean():.2f} | " + " ".join(parts)

    logger.info(
        "Interpolation sampling stats [%s] (n=%d): %s; %s; %s",
        phase_name,
        n,
        _hist("past", pasts),
        _hist("gap", gaps),
        _hist("future", futures),
    )


class AISInterpolationDataset(Dataset):
    """Sample windows made of past, gap, and future trajectory segments."""

    def __init__(
        self,
        l_data,
        max_seqlen=120,
        min_past_points=1,
        max_past_points=40,
        min_future_points=1,
        max_future_points=40,
        min_gap_points=1,
        max_gap_points=40,
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
        self.min_gap_points = min_gap_points
        self.max_gap_points = max_gap_points
        self.edge_case_prob = edge_case_prob
        self.samples_per_track = samples_per_track
        self.seed = seed
        # self.config = config  # No longer needed for global normalization

        self.min_total_points = (
            self.min_past_points + self.min_gap_points + self.min_future_points
        )
        self.l_data = [
            V for V in l_data if len(V["traj"]) >= self.min_total_points
        ]
        self.sample_refs = [
            (track_idx, sample_idx)
            for track_idx in range(len(self.l_data))
            for sample_idx in range(self.samples_per_track)
        ]

    def __len__(self):
        return len(self.sample_refs)

    def _make_rng(self, idx):
        track_idx, sample_idx = self.sample_refs[idx]
        return np.random.default_rng(self.seed + 1000003 * track_idx + sample_idx)

    def _sample_lengths(self, track_len, rng):
        max_window = min(track_len, self.max_seqlen)
        use_edge_case = (
            rng.random() < self.edge_case_prob
            and max_window >= self.min_gap_points + 2
        )

        if use_edge_case:
            past_len = 1
            future_len = 1
            gap_upper = min(self.max_gap_points, max_window - past_len - future_len)
            gap_len = int(rng.integers(self.min_gap_points, gap_upper + 1))
            return past_len, gap_len, future_len

        gap_upper = min(
            self.max_gap_points,
            max_window - self.min_past_points - self.min_future_points,
        )
        gap_len = int(rng.integers(self.min_gap_points, gap_upper + 1))

        past_upper = min(
            self.max_past_points,
            max_window - gap_len - self.min_future_points,
        )
        past_len = int(rng.integers(self.min_past_points, past_upper + 1))

        future_upper = min(
            self.max_future_points,
            max_window - gap_len - past_len,
        )
        future_len = int(rng.integers(self.min_future_points, future_upper + 1))

        return past_len, gap_len, future_len

    def __getitem__(self, idx):
        track_idx, _ = self.sample_refs[idx]
        rng = self._make_rng(idx)
        vessel = self.l_data[track_idx]
        traj = vessel["traj"].copy()
        traj[:, :5] = np.clip(traj[:, :5], 0.0, 0.9999)

        track_len = len(traj)
        past_len, orig_gap_len, future_len = self._sample_lengths(track_len, rng)
        gap_len = max(1, orig_gap_len // 2)  # Scale down by half, at least 1
        total_len = past_len + gap_len + future_len
        start_idx = int(rng.integers(0, track_len - total_len + 1))
        window = traj[start_idx:start_idx + total_len]

        seq = np.zeros((self.max_seqlen, 5), dtype=np.float32)
        seq[:total_len] = window[:, :5]

        token_types = np.zeros(self.max_seqlen, dtype=np.int64)
        token_types[:past_len] = 1
        token_types[past_len:past_len + gap_len] = 2
        token_types[past_len + gap_len:total_len] = 3

        valid_mask = np.zeros(self.max_seqlen, dtype=np.float32)
        valid_mask[:total_len] = 1.0

        target_mask = np.zeros(self.max_seqlen, dtype=np.float32)
        target_mask[past_len:past_len + gap_len] = 1.0

        time_seq = np.zeros(self.max_seqlen, dtype=np.int64)
        time_seq[:total_len] = window[:, 4].astype(np.int64)

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(token_types, dtype=torch.long),
            torch.tensor(valid_mask, dtype=torch.float32),
            torch.tensor(target_mask, dtype=torch.float32),
            torch.tensor(total_len, dtype=torch.long),
            torch.tensor(past_len, dtype=torch.long),
            torch.tensor(gap_len, dtype=torch.long),
            torch.tensor(future_len, dtype=torch.long),
            torch.tensor(vessel["mmsi"], dtype=torch.long),
            torch.tensor(time_seq, dtype=torch.long),
        )


def build_interpolation_sequence(prev_seq, next_seq, gap_len, max_seqlen):
    """Build a padded interpolation example from observed past and future tracks."""

    prev_seq = np.asarray(prev_seq, dtype=np.float32)
    next_seq = np.asarray(next_seq, dtype=np.float32)

    total_len = len(prev_seq) + gap_len + len(next_seq)
    if total_len > max_seqlen:
        raise ValueError(
            f"Interpolation window ({total_len}) exceeds max_seqlen ({max_seqlen})."
        )

    seq = np.zeros((max_seqlen, 4), dtype=np.float32)
    seq[:len(prev_seq)] = np.clip(prev_seq, 0.0, 0.9999)
    seq[len(prev_seq) + gap_len:total_len] = np.clip(next_seq, 0.0, 0.9999)

    token_types = np.zeros(max_seqlen, dtype=np.int64)
    token_types[:len(prev_seq)] = 1
    token_types[len(prev_seq):len(prev_seq) + gap_len] = 2
    token_types[len(prev_seq) + gap_len:total_len] = 3

    valid_mask = np.zeros(max_seqlen, dtype=np.float32)
    valid_mask[:total_len] = 1.0

    target_mask = np.zeros(max_seqlen, dtype=np.float32)
    target_mask[len(prev_seq):len(prev_seq) + gap_len] = 1.0

    return (
        torch.tensor(seq, dtype=torch.float32).unsqueeze(0),
        torch.tensor(token_types, dtype=torch.long).unsqueeze(0),
        torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0),
        torch.tensor(target_mask, dtype=torch.float32).unsqueeze(0),
    )
