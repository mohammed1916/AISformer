# coding=utf-8
"""Configuration flags for forecast training."""

import os

import torch


class Config:
    retrain = False
    tb_log = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_epochs = 50
    batch_size = 128
    n_samples = 4
    seed = 42

    max_seqlen = 120
    min_seqlen = 36

    min_past_points = 5
    max_past_points = 40
    min_future_points = 5
    max_future_points = 40
    edge_case_prob = 0.20

    log_gap_sampling = False
    log_gap_sample_budget = 8192

    train_samples_per_track = 4
    eval_samples_per_track = 1

    dataset_name = "ct_dma"

    if dataset_name == "ct_dma":
        lat_size = 80
        lon_size = 80
        sog_size = 10
        cog_size = 24

        n_lat_embd = 128
        n_lon_embd = 128
        n_sog_embd = 64
        n_cog_embd = 64

        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13.0
        sog_range = 30.0

    position_mode = "global_roi"
    data_lat_min = lat_min
    data_lat_max = lat_max
    data_lon_min = lon_min
    data_lon_max = lon_max
    north_km_min = -150.0
    north_km_max = 150.0
    east_km_min = -150.0
    east_km_max = 150.0
    local_origin_mode = "last_past_point"

    use_port_context = True
    port_nearest_k = 3
    port_max_distance_km = 120.0
    port_distance_scale_km = 120.0
    port_feature_size = 4
    port_context_size = port_nearest_k * port_feature_size
    port_cache_size = 200000
    port_cache_round_decimals = 3

    use_land_context = True
    land_distance_scale_km = 80.0
    land_context_size = 4
    land_cache_size = 200000
    land_cache_round_decimals = 3

    mode = "forecast"
    top_k = 10
    temperature = 1.0

    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"

    n_head = 4
    n_layer = 4
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.01
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 0

    early_stop_patience = 10
    min_improvement = 1e-4

    use_amp = True
    amp_dtype = "bfloat16"
    use_compile = True
    batch_size = 128

    filename = (
        f"{dataset_name}"
        f"-pos-{position_mode}"
        f"-forecast-past-{min_past_points}-{max_past_points}"
        f"-future-{min_future_points}-{max_future_points}"
        f"-edge-{edge_case_prob}"
        f"-samples-{train_samples_per_track}-{eval_samples_per_track}"
        f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"
        f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"
        f"-port-{int(use_port_context)}-{port_nearest_k}-{port_max_distance_km:.0f}"
        f"-land-{int(use_land_context)}-{land_distance_scale_km:.0f}"
        f"-head-{n_head}-{n_layer}"
        f"-bs-{batch_size}"
        f"-lr-{learning_rate}"
        f"-seqlen-{max_seqlen}"
    )
    savedir = "./results_forecast/" + filename + "/"
    ckpt_path = os.path.join(savedir, "model.pt")
