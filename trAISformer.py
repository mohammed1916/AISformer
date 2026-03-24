#!/usr/bin/env python
# coding: utf-8
# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

from pathlib import Path
import glob

# Optional fancy interactive menu with arrows + enter
try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False


def prompt_yes_no(question, default=False):
    default_str = "[Y/n]" if default else "[y/N]"
    while True:
        resp = input(f"{question} {default_str}: ").strip().lower()
        if resp == "" and default is not None:
            return default
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please answer y or n.")


def find_checkpoints(root_dir="."):
    print(f"Scanning for checkpoint files under '{root_dir}'...")
    p = Path(root_dir)
    ckpts = sorted(p.rglob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"Found {len(ckpts)} checkpoint(s).")
    return ckpts


def select_checkpoint():
    print("Looking for checkpoints...")
    ckpts = find_checkpoints(".")
    if not ckpts:
        print("No .pt checkpoints found in current directory tree.")
        while True:
            manual = input("Enter path to checkpoint (.pt): ").strip()
            if manual == "":
                continue
            p = Path(manual).expanduser()
            if p.is_file():
                return str(p)
            print(f"Checkpoint file not found: {p}")

    # prefer rich arrow/select UI if available
    if INQUIRER_AVAILABLE:
        choices = [Choice(value=str(p), name=f"{p} (modified: {p.stat().st_mtime:.0f})") for p in ckpts]
        choices.append(Choice(value="__custom__", name="Custom path"))
        selected = inquirer.select(message="Select checkpoint:", choices=choices, pointer="➜").execute()
        if selected != "__custom__":
            return selected
        # custom path if user selected custom
        while True:
            manual = input("Enter path to checkpoint (.pt): ").strip()
            p = Path(manual).expanduser()
            if p.is_file():
                return str(p)
            print(f"Checkpoint file not found: {p}")

    print("Found checkpoint candidates:")
    for i, p in enumerate(ckpts):
        print(f" [{i}] {p} (modified: {p.stat().st_mtime:.0f})")
    print(" [c] custom path")

    while True:
        sel = input("Select checkpoint index (or c for custom): ").strip().lower()
        if sel == "c":
            manual = input("Enter path to checkpoint (.pt): ").strip()
            p = Path(manual).expanduser()
            if p.is_file():
                return str(p)
            print(f"Checkpoint file not found: {p}")
            continue
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(ckpts):
                return str(ckpts[idx])
        print("Invalid selection, try again.")


if __name__ == "__main__":

    print("\n== TrAISformer interactive checkpoint selector ==")
    selected_ckpt = select_checkpoint()
    print(f"Using checkpoint: {selected_ckpt}")
    cf.ckpt_path = selected_ckpt

    retrain_decision = prompt_yes_no("Retrain model before evaluation?", default=False)
    cf.retrain = retrain_decision

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 4

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")

    # Yeah, done!!!
