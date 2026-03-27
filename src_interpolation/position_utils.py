# coding=utf-8
"""Helpers for location-independent interpolation position encoding."""

from __future__ import annotations

import math

import numpy as np
import torch


EARTH_RADIUS_KM = 6371.0088
_EPS = 1e-6


def clip01(x):
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 0.9999)


def uses_local_position_frame(config) -> bool:
    return getattr(config, "position_mode", "global_roi") == "local_offset_km"


def get_source_bounds(config) -> tuple[float, float, float, float]:
    lat_min = getattr(config, "data_lat_min", getattr(config, "lat_min", None))
    lat_max = getattr(config, "data_lat_max", getattr(config, "lat_max", None))
    lon_min = getattr(config, "data_lon_min", getattr(config, "lon_min", None))
    lon_max = getattr(config, "data_lon_max", getattr(config, "lon_max", None))
    if None in (lat_min, lat_max, lon_min, lon_max):
        raise ValueError("Source trajectory bounds are required for interpolation position transforms.")
    return float(lat_min), float(lat_max), float(lon_min), float(lon_max)


def get_local_bounds(config) -> tuple[float, float, float, float]:
    north_min = getattr(config, "north_km_min", None)
    north_max = getattr(config, "north_km_max", None)
    east_min = getattr(config, "east_km_min", None)
    east_max = getattr(config, "east_km_max", None)
    if None in (north_min, north_max, east_min, east_max):
        raise ValueError("Local-frame position bounds are required for local interpolation models.")
    return float(north_min), float(north_max), float(east_min), float(east_max)


def normalize_interval_np(x, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        raise ValueError("Position normalization upper bound must exceed lower bound.")
    return clip01((np.asarray(x, dtype=np.float64) - lo) / (hi - lo))


def denormalize_interval_np(x, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        raise ValueError("Position denormalization upper bound must exceed lower bound.")
    return np.asarray(x, dtype=np.float64) * (hi - lo) + lo


def denormalize_interval_torch(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    if hi <= lo:
        raise ValueError("Position denormalization upper bound must exceed lower bound.")
    return x * (hi - lo) + lo


def source_positions_to_real_np(
    lat_norm: np.ndarray,
    lon_norm: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = get_source_bounds(config)
    lat = denormalize_interval_np(lat_norm, lat_min, lat_max)
    lon = denormalize_interval_np(lon_norm, lon_min, lon_max)
    return lat.astype(np.float32), lon.astype(np.float32)


def real_positions_to_source_norm_np(
    lats: np.ndarray,
    lons: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = get_source_bounds(config)
    lat = normalize_interval_np(lats, lat_min, lat_max)
    lon = normalize_interval_np(lons, lon_min, lon_max)
    return lat.astype(np.float32), lon.astype(np.float32)


def real_to_local_offsets_np(
    lats: np.ndarray,
    lons: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.radians(np.asarray(lats, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lons, dtype=np.float64))
    origin_lat_rad = math.radians(float(origin_lat))
    origin_lon_rad = math.radians(float(origin_lon))
    north_km = (lat_rad - origin_lat_rad) * EARTH_RADIUS_KM
    east_km = (lon_rad - origin_lon_rad) * EARTH_RADIUS_KM * max(math.cos(origin_lat_rad), _EPS)
    return north_km.astype(np.float32), east_km.astype(np.float32)


def local_offsets_to_real_np(
    north_km: np.ndarray,
    east_km: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin_lat_rad = math.radians(float(origin_lat))
    lat = float(origin_lat) + np.degrees(np.asarray(north_km, dtype=np.float64) / EARTH_RADIUS_KM)
    lon = float(origin_lon) + np.degrees(
        np.asarray(east_km, dtype=np.float64) / (EARTH_RADIUS_KM * max(math.cos(origin_lat_rad), _EPS))
    )
    return lat.astype(np.float32), lon.astype(np.float32)


def real_positions_to_model_norm_np(
    lats: np.ndarray,
    lons: np.ndarray,
    config,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if lats.size == 0 or lons.size == 0:
        raise ValueError("Cannot encode an empty position sequence.")

    if uses_local_position_frame(config):
        if origin_lat is None or origin_lon is None:
            origin_lat = float(lats[-1])
            origin_lon = float(lons[-1])
        north_km, east_km = real_to_local_offsets_np(lats, lons, float(origin_lat), float(origin_lon))
        north_min, north_max, east_min, east_max = get_local_bounds(config)
        lat_norm = normalize_interval_np(north_km, north_min, north_max)
        lon_norm = normalize_interval_np(east_km, east_min, east_max)
        return lat_norm.astype(np.float32), lon_norm.astype(np.float32), float(origin_lat), float(origin_lon)

    lat_norm, lon_norm = real_positions_to_source_norm_np(lats, lons, config)
    return lat_norm, lon_norm, float(origin_lat or 0.0), float(origin_lon or 0.0)


def model_norm_to_real_np(
    lat_norm: np.ndarray,
    lon_norm: np.ndarray,
    config,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if uses_local_position_frame(config):
        if origin_lat is None or origin_lon is None:
            raise ValueError("A local-frame interpolation model requires an origin for denormalization.")
        north_min, north_max, east_min, east_max = get_local_bounds(config)
        north_km = denormalize_interval_np(lat_norm, north_min, north_max)
        east_km = denormalize_interval_np(lon_norm, east_min, east_max)
        return local_offsets_to_real_np(north_km, east_km, float(origin_lat), float(origin_lon))

    return source_positions_to_real_np(lat_norm, lon_norm, config)


def model_norm_to_real_torch(
    pos_xy: torch.Tensor,
    config,
    origin_lats: torch.Tensor | None = None,
    origin_lons: torch.Tensor | None = None,
) -> torch.Tensor:
    if uses_local_position_frame(config):
        if origin_lats is None or origin_lons is None:
            raise ValueError("A local-frame interpolation model requires batched origins for denormalization.")
        north_min, north_max, east_min, east_max = get_local_bounds(config)
        north_km = denormalize_interval_torch(pos_xy[..., 0], north_min, north_max)
        east_km = denormalize_interval_torch(pos_xy[..., 1], east_min, east_max)

        if origin_lats.dim() == 1:
            origin_lats = origin_lats.unsqueeze(-1)
        if origin_lons.dim() == 1:
            origin_lons = origin_lons.unsqueeze(-1)

        lat = origin_lats + torch.rad2deg(north_km / EARTH_RADIUS_KM)
        cos_lat = torch.cos(origin_lats * torch.pi / 180.0).clamp_min(_EPS)
        lon = origin_lons + torch.rad2deg(east_km / (EARTH_RADIUS_KM * cos_lat))
        return torch.stack((lat, lon), dim=-1)

    lat_min, lat_max, lon_min, lon_max = get_source_bounds(config)
    lat = denormalize_interval_torch(pos_xy[..., 0], lat_min, lat_max)
    lon = denormalize_interval_torch(pos_xy[..., 1], lon_min, lon_max)
    return torch.stack((lat, lon), dim=-1)


def encode_window_to_model_space(
    window: np.ndarray,
    past_len: int,
    config,
) -> tuple[np.ndarray, float, float]:
    seq = np.asarray(window[:, :4], dtype=np.float32).copy()
    seq = np.clip(seq, 0.0, 0.9999)
    if not uses_local_position_frame(config):
        return seq, 0.0, 0.0

    if past_len < 1:
        raise ValueError("Interpolation windows require at least one past point for local-frame encoding.")

    real_lats, real_lons = source_positions_to_real_np(seq[:, 0], seq[:, 1], config)
    origin_lat = float(real_lats[past_len - 1])
    origin_lon = float(real_lons[past_len - 1])
    lat_norm, lon_norm, _, _ = real_positions_to_model_norm_np(
        real_lats,
        real_lons,
        config,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    seq[:, 0] = lat_norm
    seq[:, 1] = lon_norm
    return seq, origin_lat, origin_lon
