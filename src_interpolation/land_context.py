# coding=utf-8
"""Land and coastline numeric context features for interpolation models."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points


EARTH_RADIUS_KM = 6371.0088
_EPS = 1e-6
_OBSERVED_TOKEN_IDS = (1, 3)
_DEFAULT_LAND_ROOT = Path(__file__).resolve().parents[1] / "data" / "land"
_FALLBACK_LAND_ROOT = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "marine_backend"
    / "dataset"
    / "land"
)


class LandContextEncoder:
    """Encode land/coast proximity as compact numeric features per observed AIS token."""

    def __init__(
        self,
        land_root: str | Path | None = None,
        distance_scale_km: float = 80.0,
        cache_size: int = 200000,
        cache_round_decimals: int = 3,
        use_gpu: bool = False,
    ):
        self.land_root = self._resolve_land_root(land_root)
        self.distance_scale_km = float(distance_scale_km or 1.0)
        self.cache_round_decimals = max(0, int(cache_round_decimals))
        self.context_size = 4
        self.use_gpu = use_gpu

        land_gdf = self._load(self.land_root)
        self.land_union = land_gdf.unary_union
        self.coastline = self.land_union.boundary
        # Load rasterized land mask for fast on-land checks
        import os
        land_mask_path = os.path.join(self.land_root, "land_mask.npy")
        if os.path.isfile(land_mask_path):
            self.land_mask = np.load(land_mask_path)
            self.lat_min, self.lat_max = 49.5, 62.0
            self.lon_min, self.lon_max = -5.0, 28.0
            self.grid_res = 0.01
            if self.use_gpu:
                try:
                    import cupy as cp
                    self.land_mask_gpu = cp.asarray(self.land_mask)
                except ImportError:
                    self.land_mask_gpu = None
            else:
                self.land_mask_gpu = None
        else:
            self.land_mask = None
            self.land_mask_gpu = None
        # Sample coastline as dense array of points for vectorized distance computations.
        self.coastline_points = None
        self.coastline_array = None
        if not self.coastline.is_empty:
            try:
                n_samples = 1000
                if hasattr(self.coastline, 'interpolate'):
                    coast_points = [
                        self.coastline.interpolate(float(i) / n_samples, normalized=True)
                        for i in range(n_samples)
                    ]
                    coast_np = np.asarray([[pt.y, pt.x] for pt in coast_points], dtype=np.float32)
                else:
                    # Fallback: extract coordinates from linestring/multilinestring directly.
                    coords = np.asarray(self.coastline.coords, dtype=np.float32)
                    coast_np = coords[:, ::-1] if coords.shape[1] >= 2 else None
                if coast_np is not None and coast_np.size > 0:
                    self.coastline_points = coast_np
                    if self.use_gpu:
                        try:
                            import cupy as cp
                            self.coastline_array = cp.asarray(coast_np, dtype=cp.float32)
                        except ImportError:
                            self.coastline_array = None
                else:
                    self.coastline_points = None
                    self.coastline_array = None
            except Exception:
                self.coastline_points = None
                self.coastline_array = None

        self._encode_cached = lru_cache(maxsize=max(1, int(cache_size)))(
            self._encode_single_uncached
        )

    @classmethod
    def from_config(cls, config) -> "LandContextEncoder":
        land_root = getattr(config, "land_data_root", None)
        return cls(
            land_root=land_root,
            distance_scale_km=getattr(config, "land_distance_scale_km", 80.0),
            cache_size=getattr(config, "land_cache_size", 200000),
            cache_round_decimals=getattr(config, "land_cache_round_decimals", 3),
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
            raise ValueError("Land context lat/lon arrays must have the same shape.")

        out = np.zeros((lats.size, self.context_size), dtype=np.float32)
        if lats.size == 0:
            return out

        observed_mask = np.ones(lats.shape, dtype=bool)
        if token_types is not None:
            token_types = np.asarray(token_types).reshape(-1)
            if token_types.shape[0] != lats.shape[0]:
                raise ValueError("token_types length must match land context positions.")
            observed_mask = np.isin(token_types, _OBSERVED_TOKEN_IDS)

        if self.use_gpu and self.coastline_array is not None:
            try:
                import cupy as cp
                return self._encode_positions_cupy(lats, lons, observed_mask, cp)
            except ImportError:
                print("CuPy not available, falling back to CPU/NumPy.")

        if not self.use_gpu and self.coastline_points is not None:
            return self._encode_positions_numpy(lats, lons, observed_mask)

        for idx in np.flatnonzero(observed_mask):
            lat_key = round(float(lats[idx]), self.cache_round_decimals)
            lon_key = round(float(lons[idx]), self.cache_round_decimals)
            out[idx] = np.asarray(self._encode_cached(lat_key, lon_key), dtype=np.float32)

        return out

    def _encode_positions_cupy(self, lats, lons, observed_mask, cp):
        obs_idx = np.flatnonzero(observed_mask)
        if obs_idx.size == 0 or self.coastline_array is None:
            return np.zeros((lats.size, self.context_size), dtype=np.float32)
        lat_points = cp.asarray(lats[obs_idx], dtype=cp.float32)
        lon_points = cp.asarray(lons[obs_idx], dtype=cp.float32)
        points = cp.stack([lat_points, lon_points], axis=1)  # (n_obs, 2)
        # Compute all pairwise distances to coastline points
        coast = self.coastline_array  # (n_coast, 2)
        dlat = points[:, None, 0] - coast[None, :, 0]
        dlon = points[:, None, 1] - coast[None, :, 1]
        dists = cp.sqrt(dlat ** 2 + dlon ** 2)  # (n_obs, n_coast)
        min_idx = cp.argmin(dists, axis=1)
        min_dists = dists[cp.arange(points.shape[0]), min_idx]
        # For each point, get the closest coastline point
        coast_closest = coast[min_idx]
        # Compute north/east offsets (approximate, for speed)
        north_km = (cp.radians(coast_closest[:, 0] - lat_points) * EARTH_RADIUS_KM)
        east_km = (cp.radians(coast_closest[:, 1] - lon_points) * EARTH_RADIUS_KM * cp.maximum(cp.cos(cp.radians(lat_points)), _EPS))
        # On-land check using raster mask (fully vectorized)
        if self.land_mask_gpu is not None:
            lat_idx = cp.clip(((lat_points - self.lat_min) / self.grid_res).astype(cp.int32), 0, self.land_mask_gpu.shape[0] - 1)
            lon_idx = cp.clip(((lon_points - self.lon_min) / self.grid_res).astype(cp.int32), 0, self.land_mask_gpu.shape[1] - 1)
            on_land = cp.asnumpy(self.land_mask_gpu[lat_idx, lon_idx])
        else:
            on_land = np.zeros(points.shape[0], dtype=np.float32)
        signed_distance = np.where(on_land > 0.5, -min_dists.get(), min_dists.get())
        features = np.zeros((points.shape[0], 4), dtype=np.float32)
        features[:, 0] = on_land
        features[:, 1] = np.clip(north_km.get() / self.distance_scale_km, -1.0, 1.0)
        features[:, 2] = np.clip(east_km.get() / self.distance_scale_km, -1.0, 1.0)
        features[:, 3] = np.clip(signed_distance / self.distance_scale_km, -1.0, 1.0)
        out = np.zeros((lats.size, self.context_size), dtype=np.float32)
        out[obs_idx] = features
        return out

    def _encode_positions_numpy(self, lats, lons, observed_mask):
        obs_idx = np.flatnonzero(observed_mask)
        if obs_idx.size == 0 or self.coastline_points is None:
            return np.zeros((lats.size, self.context_size), dtype=np.float32)

        # On-land check via raster mask (fast) if available.
        lat_points = lats[obs_idx]
        lon_points = lons[obs_idx]
        if self.land_mask is not None:
            lat_idx = np.clip(((lat_points - self.lat_min) / self.grid_res).astype(int), 0, self.land_mask.shape[0] - 1)
            lon_idx = np.clip(((lon_points - self.lon_min) / self.grid_res).astype(int), 0, self.land_mask.shape[1] - 1)
            on_land = self.land_mask[lat_idx, lon_idx].astype(np.float32)
        else:
            on_land = np.zeros(obs_idx.size, dtype=np.float32)

        # Compute distances to pre-sampled coastline points (approximate).
        coast = self.coastline_points  # (n_coast, 2)
        dlat = lat_points[:, None] - coast[None, :, 0]
        dlon = lon_points[:, None] - coast[None, :, 1]
        dists = np.sqrt(dlat ** 2 + dlon ** 2)
        min_idx = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(obs_idx.size), min_idx]
        coast_closest = coast[min_idx]

        north_km = (np.radians(coast_closest[:, 0] - lat_points) * EARTH_RADIUS_KM)
        east_km = (
            np.radians(coast_closest[:, 1] - lon_points)
            * EARTH_RADIUS_KM
            * np.maximum(np.cos(np.radians(lat_points)), _EPS)
        )
        signed_distance = np.where(on_land > 0.5, -min_dists, min_dists)

        features = np.zeros((obs_idx.size, 4), dtype=np.float32)
        features[:, 0] = on_land
        features[:, 1] = np.clip(north_km / self.distance_scale_km, -1.0, 1.0)
        features[:, 2] = np.clip(east_km / self.distance_scale_km, -1.0, 1.0)
        features[:, 3] = np.clip(signed_distance / self.distance_scale_km, -1.0, 1.0)

        out = np.zeros((lats.size, self.context_size), dtype=np.float32)
        out[obs_idx] = features
        return out

    @staticmethod
    def _resolve_land_root(land_root: str | Path | None) -> Path:
        candidates = []
        if land_root is not None:
            candidates.append(Path(land_root).expanduser().resolve())
        candidates.extend([_DEFAULT_LAND_ROOT.resolve(), _FALLBACK_LAND_ROOT.resolve()])
        for candidate in candidates:
            main = candidate / "main" / "ne_10m_land.shp"
            minor = candidate / "minor_islands" / "ne_10m_minor_islands.shp"
            if main.is_file() and minor.is_file():
                return candidate
        raise FileNotFoundError("Land shapefiles not found under TrAISformer/data/land or backend dataset/land.")

    @staticmethod
    def _load(land_root: Path) -> gpd.GeoDataFrame:
        main_land_shp = land_root / "main" / "ne_10m_land.shp"
        minor_islands_shp = land_root / "minor_islands" / "ne_10m_minor_islands.shp"

        land_gdf_main = gpd.read_file(main_land_shp)
        land_gdf_minor = gpd.read_file(minor_islands_shp)
        land_gdf = gpd.GeoDataFrame(
            pd.concat([land_gdf_main, land_gdf_minor], ignore_index=True),
            geometry="geometry",
            crs=land_gdf_main.crs or land_gdf_minor.crs,
        )
        if land_gdf.crs is not None and str(land_gdf.crs) != "EPSG:4326":
            land_gdf = land_gdf.to_crs(epsg=4326)
        return land_gdf

    def _encode_single_uncached(self, lat: float, lon: float) -> tuple[float, ...]:
        pt = Point(float(lon), float(lat))
        on_land = 1.0 if self.land_union.contains(pt) else 0.0

        if self.coastline.is_empty:
            return (on_land, 0.0, 0.0, 0.0)

        coast_pt = nearest_points(pt, self.coastline)[1]
        north_km, east_km = self._relative_offsets_km(
            float(lat),
            float(lon),
            float(coast_pt.y),
            float(coast_pt.x),
        )
        distance_km = math.hypot(float(north_km), float(east_km))
        signed_distance = -distance_km if on_land > 0.5 else distance_km
        features = (
            on_land,
            float(np.clip(north_km / self.distance_scale_km, -1.0, 1.0)),
            float(np.clip(east_km / self.distance_scale_km, -1.0, 1.0)),
            float(np.clip(signed_distance / self.distance_scale_km, -1.0, 1.0)),
        )
        return features

    @staticmethod
    def _relative_offsets_km(
        lat: float,
        lon: float,
        coast_lat: float,
        coast_lon: float,
    ) -> tuple[float, float]:
        lat_rad = math.radians(lat)
        north_km = math.radians(coast_lat - lat) * EARTH_RADIUS_KM
        east_km = (
            math.radians(coast_lon - lon)
            * EARTH_RADIUS_KM
            * max(math.cos(lat_rad), _EPS)
        )
        return float(north_km), float(east_km)
