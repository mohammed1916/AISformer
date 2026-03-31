import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src_interpolation')))
from land_context import LandContextEncoder
import time

# Generate random points within the region
np.random.seed(42)
lats = np.random.uniform(50.6, 61.5, size=50)
lons = np.random.uniform(-4.4, 27.9, size=50)

# Instantiate with GPU acceleration
encoder_gpu = LandContextEncoder(use_gpu=True)
start_gpu = time.time()
features_gpu = encoder_gpu.encode_positions(lats, lons)
gpu_time = time.time() - start_gpu

# Instantiate with CPU (for comparison)
encoder_cpu = LandContextEncoder(use_gpu=False)
start_cpu = time.time()
features_cpu = encoder_cpu.encode_positions(lats, lons)
cpu_time = time.time() - start_cpu

print("GPU features (first 2 rows):\n", features_gpu[:2])
print("CPU features (first 2 rows):\n", features_cpu[:2])
print("Max absolute difference:", np.abs(features_gpu - features_cpu).max())
print(f"GPU time: {gpu_time:.4f}s, CPU time: {cpu_time:.4f}s")
