# coding=utf-8
"""Models for trajectory interpolation."""

import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class TrAISformerInterpolation(nn.Module):
    """Bidirectional transformer that infills a masked middle trajectory span."""

    PADDING_SEGMENT_ID = 0
    PAST_SEGMENT_ID = 1
    GAP_SEGMENT_ID = 2
    FUTURE_SEGMENT_ID = 3

    def __init__(self, config):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.heading_size = config.heading_size
        self.full_size = config.full_size
        self.max_seqlen = config.max_seqlen

        self.register_buffer(
            "att_sizes",
            torch.tensor(
                [self.lat_size, self.lon_size, self.sog_size, self.cog_size, self.heading_size],
                dtype=torch.float32,
            ),
        )
        self.mask_token_ids = (
            self.lat_size,
            self.lon_size,
            self.sog_size,
            self.cog_size,
            self.heading_size,
        )

        self.lat_emb = nn.Embedding(self.lat_size + 1, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size + 1, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size + 1, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size + 1, config.n_cog_embd)
        self.heading_emb = nn.Embedding(self.heading_size + 1, config.n_heading_embd)
        self.segment_emb = nn.Embedding(4, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.resid_pdrop,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layer,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.full_size, bias=False)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        return torch.optim.AdamW(
            self.parameters(),
            lr=train_config.learning_rate,
            betas=train_config.betas,
            weight_decay=train_config.weight_decay,
        )

    def to_indexes(self, x):
        x = x.clamp(0.0, 0.9999)
        return (x * self.att_sizes.view(1, 1, -1)).long()

    def _masked_inputs(self, idxs, token_types):
        inputs = idxs.clone()
        gap_positions = token_types == self.GAP_SEGMENT_ID
        pad_positions = token_types == self.PADDING_SEGMENT_ID

        for dim, mask_id in enumerate(self.mask_token_ids):
            inputs[:, :, dim] = torch.where(
                gap_positions,
                torch.full_like(inputs[:, :, dim], mask_id),
                inputs[:, :, dim],
            )
            inputs[:, :, dim] = torch.where(
                pad_positions,
                torch.zeros_like(inputs[:, :, dim]),
                inputs[:, :, dim],
            )

        return inputs

    def forward(
        self,
        x,
        token_types,
        valid_mask=None,
        target_mask=None,
        with_targets=False,
    ):
        idxs = self.to_indexes(x)
        inputs = self._masked_inputs(idxs, token_types)

        lat_embeddings = self.lat_emb(inputs[:, :, 0])
        lon_embeddings = self.lon_emb(inputs[:, :, 1])
        sog_embeddings = self.sog_emb(inputs[:, :, 2])
        cog_embeddings = self.cog_emb(inputs[:, :, 3])
        heading_embeddings = self.heading_emb(inputs[:, :, 4])
        token_embeddings = torch.cat(
            (lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings, heading_embeddings),
            dim=-1,
        )

        batchsize, seqlen, _ = token_embeddings.size()
        if seqlen > self.max_seqlen:
            raise ValueError("Input sequence is longer than max_seqlen.")

        hidden = token_embeddings + self.pos_emb[:, :seqlen, :] + self.segment_emb(token_types)
        hidden = self.drop(hidden)
        padding_mask = None if valid_mask is None else ~valid_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.ln_f(hidden)
        logits = self.head(hidden)

        loss = None
        if with_targets:
            lat_logits, lon_logits, sog_logits, cog_logits, heading_logits = torch.split(
                logits,
                (self.lat_size, self.lon_size, self.sog_size, self.cog_size, self.heading_size),
                dim=-1,
            )

            lat_loss = F.cross_entropy(
                lat_logits.reshape(-1, self.lat_size),
                idxs[:, :, 0].reshape(-1),
                reduction="none",
            ).view(batchsize, seqlen)
            lon_loss = F.cross_entropy(
                lon_logits.reshape(-1, self.lon_size),
                idxs[:, :, 1].reshape(-1),
                reduction="none",
            ).view(batchsize, seqlen)
            sog_loss = F.cross_entropy(
                sog_logits.reshape(-1, self.sog_size),
                idxs[:, :, 2].reshape(-1),
                reduction="none",
            ).view(batchsize, seqlen)
            cog_loss = F.cross_entropy(
                cog_logits.reshape(-1, self.cog_size),
                idxs[:, :, 3].reshape(-1),
                reduction="none",
            ).view(batchsize, seqlen)
            heading_loss = F.cross_entropy(
                heading_logits.reshape(-1, self.heading_size),
                idxs[:, :, 4].reshape(-1),
                reduction="none",
            ).view(batchsize, seqlen)

            loss = lat_loss + lon_loss + sog_loss + cog_loss + heading_loss
            if target_mask is not None:
                denom = target_mask.sum(dim=1).clamp_min(1.0)
                loss = (loss * target_mask).sum(dim=1) / denom
            elif valid_mask is not None:
                denom = valid_mask.sum(dim=1).clamp_min(1.0)
                loss = (loss * valid_mask).sum(dim=1) / denom
            loss = loss.mean()

        return logits, loss
