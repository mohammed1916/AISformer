from src_interpolation.config_trAISformer import Config

# Add presets keyed by checkpoint style or explicit name.
CKPT_PRESETS = {
    "ct_dma-interp-gap-1-40-past-1-40-future-1-40-edge-0.2-samples-4-1-data_size-250-270-30-72-embd_size-256-256-128-128-head-8-8-bs-32-lr-0.0006-seqlen-120": {
        "lat_size": 250,
        "lon_size": 270,
        "sog_size": 30,
        "cog_size": 72,
        "n_lat_embd": 256,
        "n_lon_embd": 256,
        "n_sog_embd": 128,
        "n_cog_embd": 128,
        "n_head": 8,
        "n_layer": 8,
        "max_seqlen": 120,
        "use_port_context": False,
        "port_context_size": 0,
        "use_land_context": False,
        "land_context_size": 0,
    }
}


def apply_checkpoint_config(config, checkpoint_path):
    name = checkpoint_path.replace("\\", "/").split("/")[-2]
    if name in CKPT_PRESETS:
        preset = CKPT_PRESETS[name]
        for k, v in preset.items():
            setattr(config, k, v)
        # recompute derived values
        config.full_size = config.lat_size + config.lon_size + config.sog_size + config.cog_size
        config.n_embd = config.n_lat_embd + config.n_lon_embd + config.n_sog_embd + config.n_cog_embd
        return config
    raise ValueError(f"No preset for checkpoint directory {name}")
