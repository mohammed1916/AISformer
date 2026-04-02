import sys
import time
import torch

# Ensure local modules resolve
sys.path.insert(0, ".")
sys.path.insert(0, "src_interpolation")

from src_interpolation.config_trAISformer import Config
from src_interpolation.datasets import build_interpolation_sequence
from src_interpolation import models, trainers

config = Config()
config.use_port_context = False
config.use_land_context = False
config.lat_size = 250
config.lon_size = 270
config.sog_size = 30
config.cog_size = 72
config.n_lat_embd = 256
config.n_lon_embd = 256
config.n_sog_embd = 128
config.n_cog_embd = 128
config.n_head = 8
config.n_layer = 8
config.max_seqlen = 120
config.full_size = config.lat_size + config.lon_size + config.sog_size + config.cog_size
config.n_embd = config.n_lat_embd + config.n_lon_embd + config.n_sog_embd + config.n_cog_embd

checkpoint = "results_interpolation/ct_dma-interp-gap-1-40-past-1-40-future-1-40-edge-0.2-samples-4-1-data_size-250-270-30-72-embd_size-256-256-128-128-head-8-8-bs-32-lr-0.0006-seqlen-120/model.pt"

model = models.TrAISformerInterpolation(config)
state = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(state)
model = model.to(config.device)
model.eval()

cases = [
    (1, 1, 1),
    (5, 5, 5),
    (10, 10, 10),
    (20, 20, 20),
    (40, 40, 40),
]

print("device", config.device)
print("model params", sum(p.numel() for p in model.parameters()))

for past_len, gap_len, future_len in cases:
    prev = torch.rand((past_len, 4), dtype=torch.float32)
    nxt = torch.rand((future_len, 4), dtype=torch.float32)

    seqs, token_types, valid_mask, target_mask, port_feats, land_feats = build_interpolation_sequence(
        prev.numpy(),
        nxt.numpy(),
        gap_len=gap_len,
        max_seqlen=config.max_seqlen,
        port_encoder=None,
        land_encoder=None,
        prev_real_points=prev.numpy(),
        next_real_points=nxt.numpy(),
        port_context_size=0,
        land_context_size=0,
    )

    seqs = seqs.to(config.device)
    token_types = token_types.to(config.device)
    valid_mask = valid_mask.to(config.device)

    # Warmup
    for _ in range(5):
        _ = trainers.predict_gap(model, seqs, token_types, valid_mask)
    if config.device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    n_iter = 50
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = trainers.predict_gap(model, seqs, token_types, valid_mask)
    if config.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(
        f"case past={past_len} gap={gap_len} future={future_len} -> avg={elapsed_ms/n_iter:.3f} ms (total {elapsed_ms:.3f} ms over {n_iter})"
    )
