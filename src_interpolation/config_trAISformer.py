# coding=utf-8
"""Configuration flags for interpolation training."""

import os
import torch


class Config:
    retrain = True
    tb_log = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_epochs = 50
    batch_size = 32
    n_samples = 4
    seed = 42

    max_seqlen = 120
    min_seqlen = 36

    min_past_points = 1
    max_past_points = 40
    min_future_points = 1
    max_future_points = 40
    min_gap_points = 1
    max_gap_points = 40
    edge_case_prob = 0.20

    train_samples_per_track = 4
    eval_samples_per_track = 1

    dataset_name = "ct_dma"

    if dataset_name == "ct_dma":
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13.0
        sog_range = 30.0

    mode = "interpolation"
    top_k = 10
    temperature = 1.0

    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"

    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 260e9
    num_workers = 4

    filename = (
        f"{dataset_name}"
        f"-interp-gap-{min_gap_points}-{max_gap_points}"
        f"-past-{min_past_points}-{max_past_points}"
        f"-future-{min_future_points}-{max_future_points}"
        f"-edge-{edge_case_prob}"
        f"-samples-{train_samples_per_track}-{eval_samples_per_track}"
        f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"
        f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"
        f"-head-{n_head}-{n_layer}"
        f"-bs-{batch_size}"
        f"-lr-{learning_rate}"
        f"-seqlen-{max_seqlen}"
    )
    savedir = "./results_interpolation/" + filename + "/"

    ckpt_path = os.path.join(savedir, "model.pt")
