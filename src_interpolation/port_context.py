# coding=utf-8
"""Nearby-port numeric context features for interpolation models."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
_EPS = 1e-6
_OBSERVED_TOKEN_IDS = (1, 3)
_DEFAULT_PORT_CSV = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "marine_backend"
    / "dataset"
    / "port"
    / "UpdatedPub150.csv"
)


class PortContextEncoder:
    """Encode nearby ports as compact numeric features per observed AIS token."""

    def __init__(
        self,
        csv_path: str | Path = _DEFAULT_PORT_CSV,
        nearest_k: int = 3,
        max_distance_km: float = 120.0,
        distance_scale_km: float | None = None,
        cache_size: int = 200000,
        cache_round_decimals: int = 3,
    ):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.nearest_k = max(0, int(nearest_k))
        self.max_distance_km = float(max_distance_km)
        self.distance_scale_km = float(distance_scale_km or max_distance_km or 1.0)
        self.cache_round_decimals = max(0, int(cache_round_decimals))
        self.context_size = self.nearest_k * 4

        df = self._load(self.csv_path)
        self.port_lats = df["lat"].to_numpy(dtype=np.float64)
        self.port_lons = df["lon"].to_numpy(dtype=np.float64)

        self._encode_cached = lru_cache(maxsize=max(1, int(cache_size)))(
            self._encode_single_uncached
        )

    @classmethod
    def from_config(cls, config) -> "PortContextEncoder":
        csv_path = getattr(config, "port_csv_path", _DEFAULT_PORT_CSV)
        return cls(
            csv_path=csv_path,
            nearest_k=getattr(config, "port_nearest_k", 3),
            max_distance_km=getattr(config, "port_max_distance_km", 120.0),
            distance_scale_km=getattr(config, "port_distance_scale_km", None),
            cache_size=getattr(config, "port_cache_size", 200000),
            cache_round_decimals=getattr(config, "port_cache_round_decimals", 3),
        )

    def encode_positions(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        token_types: np.ndarray | None = None,
    ) -> np.ndarray:
        lats = np.asarray(lats, dtype=np.float64).reshape(-1)
        lons = np.asarray(lons, dtype=np.float64).reshape(-1)
        if lats.shape != lons.shape:
            raise ValueError("Port context lat/lon arrays must have the same shape.")

        out = np.zeros((lats.size, self.context_size), dtype=np.float32)
        if self.context_size == 0 or lats.size == 0:
            return out

        observed_mask = np.ones(lats.shape, dtype=bool)
        if token_types is not None:
            token_types = np.asarray(token_types).reshape(-1)
            if token_types.shape[0] != lats.shape[0]:
                raise ValueError("token_types length must match port context positions.")
            observed_mask = np.isin(token_types, _OBSERVED_TOKEN_IDS)

        for idx in np.flatnonzero(observed_mask):
            lat_key = round(float(lats[idx]), self.cache_round_decimals)
            lon_key = round(float(lons[idx]), self.cache_round_decimals)
            out[idx] = np.asarray(self._encode_cached(lat_key, lon_key), dtype=np.float32)

        return out

    @staticmethod
    def _load(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df = df.dropna(subset=["Latitude", "Longitude"]).copy()
        df["lat"] = df["Latitude"].astype(float)
        df["lon"] = df["Longitude"].astype(float)
        return df

    def _encode_single_uncached(self, lat: float, lon: float) -> tuple[float, ...]:
        if self.context_size == 0 or self.port_lats.size == 0:
            return tuple()

        dists = self._haversine_km(float(lat), float(lon))
        within = np.flatnonzero(dists <= self.max_distance_km)
        features = np.zeros((self.nearest_k, 4), dtype=np.float32)
        if within.size == 0:
            return tuple(features.reshape(-1).tolist())

        if within.size > self.nearest_k:
            nearest_local = np.argpartition(dists[within], self.nearest_k - 1)[: self.nearest_k]
            nearest = within[nearest_local]
        else:
            nearest = within
        nearest = nearest[np.argsort(dists[nearest])]

        sel_lats = self.port_lats[nearest]
        sel_lons = self.port_lons[nearest]
        sel_dists = dists[nearest]
        north_km, east_km = self._relative_offsets_km(float(lat), float(lon), sel_lats, sel_lons)

        count = min(nearest.size, self.nearest_k)
        features[:count, 0] = 1.0
        features[:count, 1] = np.clip(north_km[:count] / self.distance_scale_km, -1.0, 1.0)
        features[:count, 2] = np.clip(east_km[:count] / self.distance_scale_km, -1.0, 1.0)
        features[:count, 3] = np.clip(sel_dists[:count] / self.distance_scale_km, 0.0, 1.0)
        return tuple(features.reshape(-1).tolist())

    def _haversine_km(self, lat: float, lon: float) -> np.ndarray:
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        lat2 = np.radians(self.port_lats)
        lon2 = np.radians(self.port_lons)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
        return EARTH_RADIUS_KM * c

    @staticmethod
    def _relative_offsets_km(
        lat: float,
        lon: float,
        port_lats: np.ndarray,
        port_lons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lat_rad = math.radians(lat)
        north_km = np.radians(np.asarray(port_lats, dtype=np.float64) - lat) * EARTH_RADIUS_KM
        east_km = (
            np.radians(np.asarray(port_lons, dtype=np.float64) - lon)
            * EARTH_RADIUS_KM
            * max(math.cos(lat_rad), _EPS)
        )
        return north_km.astype(np.float32), east_km.astype(np.float32)
