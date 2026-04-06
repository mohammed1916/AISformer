# coding=utf-8
"""Training utilities for interpolation models."""

import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import utils


logger = logging.getLogger(__name__)


@torch.no_grad()
def decode_logits(model, logits, sample=False, temperature=1.0, top_k=None):
    logits = logits / temperature
    lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
        logits,
        (model.lat_size, model.lon_size, model.sog_size, model.cog_size),
        dim=-1,
    )

    if top_k is not None:
        original_shape = lat_logits.shape
        lat_logits = utils.top_k_logits(lat_logits.reshape(-1, model.lat_size), min(top_k, model.lat_size)).view(original_shape)
        lon_logits = utils.top_k_logits(lon_logits.reshape(-1, model.lon_size), min(top_k, model.lon_size)).view(lon_logits.shape)
        sog_logits = utils.top_k_logits(sog_logits.reshape(-1, model.sog_size), min(top_k, model.sog_size)).view(sog_logits.shape)
        cog_logits = utils.top_k_logits(cog_logits.reshape(-1, model.cog_size), min(top_k, model.cog_size)).view(cog_logits.shape)

    lat_probs = F.softmax(lat_logits, dim=-1)
    lon_probs = F.softmax(lon_logits, dim=-1)
    sog_probs = F.softmax(sog_logits, dim=-1)
    cog_probs = F.softmax(cog_logits, dim=-1)

    if sample:
        lat_ix = torch.multinomial(lat_probs.reshape(-1, model.lat_size), num_samples=1).view(lat_probs.shape[:-1] + (1,))
        lon_ix = torch.multinomial(lon_probs.reshape(-1, model.lon_size), num_samples=1).view(lon_probs.shape[:-1] + (1,))
        sog_ix = torch.multinomial(sog_probs.reshape(-1, model.sog_size), num_samples=1).view(sog_probs.shape[:-1] + (1,))
        cog_ix = torch.multinomial(cog_probs.reshape(-1, model.cog_size), num_samples=1).view(cog_probs.shape[:-1] + (1,))
    else:
        lat_ix = torch.argmax(lat_probs, dim=-1, keepdim=True)
        lon_ix = torch.argmax(lon_probs, dim=-1, keepdim=True)
        sog_ix = torch.argmax(sog_probs, dim=-1, keepdim=True)
        cog_ix = torch.argmax(cog_probs, dim=-1, keepdim=True)

    ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
    return (ix.float() + 0.5) / model.att_sizes.view(1, 1, -1)


@torch.no_grad()
def predict_gap(
    model,
    seqs,
    token_types,
    valid_masks,
    port_context=None,
    land_context=None,
    sample=False,
    temperature=1.0,
    top_k=None,
):
    model.eval()
    logits, _ = model(
        seqs,
        token_types=token_types,
        valid_mask=valid_masks,
        port_context=port_context,
        land_context=land_context,
        with_targets=False,
    )
    decoded = decode_logits(model, logits, sample=sample, temperature=temperature, top_k=top_k)
    completed = seqs.clone()
    gap_mask = token_types == model.GAP_SEGMENT_ID
    completed[gap_mask] = decoded[gap_mask]
    return completed


class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, config, savedir=None, device=torch.device("cpu"), aisdls=None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.savedir = savedir
        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls or {}

        # AMP: use BF16 on CUDA (Ada Lovelace sm_89 has native BF16 Tensor Cores)
        # GradScaler is for FP16 only; BF16 has FP32-level dynamic range, no scaling needed
        self._amp_enabled = torch.cuda.is_available() and getattr(config, "use_amp", True)
        self._amp_dtype = torch.bfloat16 if getattr(config, "amp_dtype", "bfloat16") == "bfloat16" else torch.float16
        self._scaler = torch.amp.GradScaler("cuda", enabled=(self._amp_enabled and self._amp_dtype == torch.float16))
        if self._amp_enabled:
            logging.info("AMP enabled: dtype=%s", self._amp_dtype)

    def save_checkpoint(self, best_epoch):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # Unwrap compiled model if needed
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def _plot_predictions(self, epoch):
        if "test" not in self.aisdls:
            return

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        (
            seqs,
            token_types,
            valid_masks,
            target_masks,
            seqlens,
            past_lens,
            gap_lens,
            future_lens,
            mmsis,
            time_seqs,
            port_features,
            land_features,
        ) = next(iter(self.aisdls["test"]))
        n_plots = min(6, seqs.shape[0])
        seqs = seqs[:n_plots].to(self.device, non_blocking=True)
        token_types = token_types[:n_plots].to(self.device, non_blocking=True)
        valid_masks = valid_masks[:n_plots].to(self.device, non_blocking=True)
        port_features = port_features[:n_plots].to(self.device, non_blocking=True)
        land_features = land_features[:n_plots].to(self.device, non_blocking=True)

        preds = predict_gap(
            raw_model,
            seqs,
            token_types,
            valid_masks,
            port_context=port_features,
            land_context=land_features,
            sample=False,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
        )

        inputs_np = seqs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        token_types_np = token_types.detach().cpu().numpy()

        img_path = os.path.join(self.savedir, f"epoch_{epoch + 1:03d}.jpg")
        plt.figure(figsize=(9, 6), dpi=150)
        cmap = plt.cm.get_cmap("tab10")

        # Resolve segment IDs from underlying model (may be torch.compile wrapped)
        _m = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        for idx in range(n_plots):
            c = cmap(float(idx) / max(1, n_plots - 1))
            tt = token_types_np[idx]

            past_mask = tt == _m.PAST_SEGMENT_ID
            gap_mask = tt == _m.GAP_SEGMENT_ID
            future_mask = tt == _m.FUTURE_SEGMENT_ID

            plt.plot(inputs_np[idx][past_mask, 1], inputs_np[idx][past_mask, 0], "-o", color=c, linewidth=1.5, markersize=3)
            plt.plot(inputs_np[idx][future_mask, 1], inputs_np[idx][future_mask, 0], "-o", color=c, linewidth=1.5, markersize=3)
            plt.plot(inputs_np[idx][gap_mask, 1], inputs_np[idx][gap_mask, 0], "--", color=c, linewidth=1.0, alpha=0.6)
            plt.plot(preds_np[idx][gap_mask, 1], preds_np[idx][gap_mask, 0], "x", color=c, markersize=4)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("Past/Future-conditioned gap predictions")
        plt.savefig(img_path, dpi=150)
        plt.close()

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        _orig = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        optimizer = _orig.configure_optimizers(config)

        # pin_memory=True works with num_workers=0 in PyTorch 2.x and allows
        # non_blocking=True GPU transfers to overlap with CPU work
        use_pin = torch.cuda.is_available()

        def run_epoch(split, epoch=0):
            is_train = split == "Training"
            model.train(is_train)
            data = self.train_dataset if is_train else self.valid_dataset
            loader = DataLoader(
                data,
                shuffle=is_train,
                pin_memory=use_pin,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            running_loss, n_items = 0.0, 0

            for it, batch in pbar:
                (
                    seqs,
                    token_types,
                    valid_masks,
                    target_masks,
                    seqlens,
                    past_lens,
                    gap_lens,
                    future_lens,
                    mmsis,
                    time_seqs,
                    port_features,
                    land_features,
                ) = batch
                # non_blocking=True: overlaps H2D DMA with CPU preprocessing
                seqs = seqs.to(self.device, non_blocking=True)
                token_types = token_types.to(self.device, non_blocking=True)
                valid_masks = valid_masks.to(self.device, non_blocking=True)
                target_masks = target_masks.to(self.device, non_blocking=True)
                port_features = port_features.to(self.device, non_blocking=True)
                land_features = land_features.to(self.device, non_blocking=True)

                # BF16 autocast: Tensor Cores on sm_89 run BF16 at native speed.
                # Forward + loss are cast; backward is kept in FP32 (PyTorch handles this).
                amp_ctx = torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._amp_enabled)
                with torch.set_grad_enabled(is_train), amp_ctx:
                    _, loss = model(
                        seqs,
                        token_types=token_types,
                        valid_mask=valid_masks,
                        target_mask=target_masks,
                        port_context=port_features,
                        land_context=land_features,
                        with_targets=True,
                    )
                    loss = loss.mean()

                losses.append(loss.item())
                running_loss += loss.item() * seqs.shape[0]
                n_items += seqs.shape[0]

                if is_train:
                    model.zero_grad(set_to_none=True)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self._scaler.step(optimizer)
                    self._scaler.update()

                    if config.lr_decay:
                        self.tokens += target_masks.sum().item()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

            mean_loss = running_loss / max(1, n_items)
            logging.info(f"{split}, epoch {epoch + 1}, loss {mean_loss:.5f}.")
            return float(np.mean(losses))

        best_loss = float("inf")
        best_epoch = 0
        no_improve_epochs = 0
        self.tokens = 0

        for epoch in range(config.max_epochs):
            train_loss = run_epoch("Training", epoch=epoch)
            valid_loss = run_epoch("Valid", epoch=epoch) if self.valid_dataset is not None else None

            logging.info(
                "Epoch %d summary: train_loss=%.5f valid_loss=%s",
                epoch + 1,
                train_loss,
                f"{valid_loss:.5f}" if valid_loss is not None else "N/A",
            )

            good_model = self.valid_dataset is None or (valid_loss is not None and valid_loss < best_loss - config.min_improvement)
            if self.config.ckpt_path is not None and good_model:
                best_loss = valid_loss if valid_loss is not None else best_loss
                best_epoch = epoch
                no_improve_epochs = 0
                self.save_checkpoint(best_epoch + 1)
            else:
                if self.valid_dataset is not None:
                    no_improve_epochs += 1

            self._plot_predictions(epoch)

            if self.valid_dataset is not None and no_improve_epochs >= getattr(config, "early_stop_patience", 10):
                logging.info(
                    "Early stopping after %d epochs without improvement (best epoch=%d, best loss=%.5f).",
                    no_improve_epochs,
                    best_epoch + 1,
                    best_loss,
                )
                break

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        _orig = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        final_epoch = epoch + 1
        logging.info(f"Last epoch: {final_epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{final_epoch:03d}.pt")
        torch.save(_orig.state_dict(), save_path)
