import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

# Parameters for rasterization
grid_res = 0.01  # degrees
lat_min, lat_max = 49.5, 62.0
lon_min, lon_max = -5.0, 28.0

# Load land polygons (same as LandContextEncoder)
land_root = os.path.join(os.path.dirname(__file__), "../data/land")
main_land_shp = os.path.join(land_root, "main", "ne_10m_land.shp")
minor_islands_shp = os.path.join(land_root, "minor_islands", "ne_10m_minor_islands.shp")
land_gdf_main = gpd.read_file(main_land_shp)
land_gdf_minor = gpd.read_file(minor_islands_shp)
land_gdf = gpd.GeoDataFrame(
    pd.concat([land_gdf_main, land_gdf_minor], ignore_index=True),
    geometry="geometry",
    crs=land_gdf_main.crs or land_gdf_minor.crs,
)
if land_gdf.crs is not None and str(land_gdf.crs) != "EPSG:4326":
    land_gdf = land_gdf.to_crs(epsg=4326)

# Create grid
lat_grid = np.arange(lat_min, lat_max, grid_res)
lon_grid = np.arange(lon_min, lon_max, grid_res)
mask = np.zeros((lat_grid.size, lon_grid.size), dtype=np.uint8)

print(f"Rasterizing land polygons to grid {mask.shape}...")
for i, lat in enumerate(lat_grid):
    for j, lon in enumerate(lon_grid):
        pt = Point(lon, lat)
        if land_gdf.contains(pt).any():
            mask[i, j] = 1

np.save(os.path.join(land_root, "land_mask.npy"), mask)
print(f"Saved land mask to {os.path.join(land_root, 'land_mask.npy')}")
