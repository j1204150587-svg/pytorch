# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure PyTorch multi-table HSTU training workload.

RMSNorm and HSTUBlock are adapted from
recsys-upstream-research/corelib/talos/examples/hstu/model.py. The original
causal SiLU attention, gating, and tied item output are retained. Independent
side-feature embeddings provide many genuine trainable parameter tensors.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import planned_rms as _planned_rms
_planned_rms.install(os.environ["NPU_RMS_LUT_PARTS_LIBRARY"])


@dataclass(frozen=True)
class ModelConfig:
    num_tables: int = 256
    table_rows: int = 256
    features_per_table: int = 4
    vocab_size: int = 1024
    seq_len: int = 32
    embedding_dim: int = 64
    num_heads: int = 4
    hidden_dim: int = 16
    attention_dim: int = 16
    num_layers: int = 4
    dropout: float = 0.0
    num_user_groups: int = 32

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "dropout" and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.training and torch.is_grad_enabled() and x.ndim == 3 and x.shape[-1] == 64
                and self.weight.shape == (64,) and x.dtype == self.weight.dtype == torch.bfloat16
                and x.device == self.weight.device and x.device.type == "npu"):
            return _planned_rms.run(x,self.weight,self.eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class HSTUBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attn_alpha = 1.0 / math.sqrt(attention_dim)

        projected_dim = 2 * num_heads * (hidden_dim + attention_dim)
        self.input_norm = RMSNorm(embedding_dim)
        self.uvqk = nn.Linear(embedding_dim, projected_dim)
        self.output_norm = RMSNorm(num_heads * hidden_dim)
        self.output = nn.Linear(num_heads * hidden_dim * 2 + embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        uvqk = self.uvqk(self.input_norm(x))
        u, v, q, k = torch.split(
            uvqk,
            [
                self.num_heads * self.hidden_dim,
                self.num_heads * self.hidden_dim,
                self.num_heads * self.attention_dim,
                self.num_heads * self.attention_dim,
            ],
            dim=-1,
        )
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = F.silu(scores).masked_fill(~causal, 0.0)
        scores = scores / float(seq_len)
        attn = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)

        gated = self.output_norm(attn) * F.silu(u)
        y = torch.cat([gated, u, residual], dim=-1)
        return residual + self.dropout(self.output(y))


class MultiTableHSTU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(config.num_user_groups + 1, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.seq_len, config.embedding_dim)
        self.side_embeddings = nn.ModuleList(
            [nn.Embedding(config.table_rows, config.embedding_dim) for _ in range(config.num_tables)]
        )
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    hidden_dim=config.hidden_dim,
                    attention_dim=config.attention_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = RMSNorm(config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.output.weight = self.item_embedding.weight
        self._side_calls = self._side_hits = 0
        if os.environ.get("NPU_DYNAMIC_GROUPED_SIDE") == "1":
            offsets = torch.arange(config.num_tables, dtype=torch.int64) * config.table_rows
            self.register_buffer("_side_offsets", offsets.view(-1, 1, 1, 1), persistent=False)

    def forward(
        self,
        item_ids: torch.Tensor,
        user_group: torch.Tensor,
        feature_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, L, V] logits from [B, L], [B], and [B, L, T, F] IDs."""
        if item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [B, L]")
        batch_size, seq_len = item_ids.shape
        if not 0 < seq_len <= self.config.seq_len:
            raise ValueError("sequence length must be between 1 and config.seq_len")
        if user_group.shape != (batch_size,):
            raise ValueError("user_group must have shape [B]")
        if feature_ids.shape != (
            batch_size,
            seq_len,
            self.config.num_tables,
            self.config.features_per_table,
        ):
            raise ValueError("feature_ids must have shape [B, L, num_tables, features_per_table]")

        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0)
        x = self.item_embedding(item_ids) + self.position_embedding(positions)
        x = x + self.user_embedding(user_group).unsqueeze(1)
        side = None
        if hasattr(self, "_side_offsets"):
            from side_tables_op import model_side_sum
            side = model_side_sum(self, feature_ids)
        if side is None:
            for table_index, table in enumerate(self.side_embeddings):
                table_features = table(feature_ids[:, :, table_index, :]).mean(dim=-2)
                side = table_features if side is None else side + table_features
        x = x + side / math.sqrt(self.config.num_tables)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def model_summary(model: MultiTableHSTU, config: ModelConfig) -> dict[str, Any]:
    """Count unique trainable weights; the tied output weight counts only once.

    embedding_table_tensors includes side, item, user, and position embeddings.
    side_embedding_elements includes only the independent side-feature tables.
    """
    trainable = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    embeddings = {
        id(module.weight): module.weight
        for module in model.modules()
        if isinstance(module, nn.Embedding) and module.weight.requires_grad
    }
    side_embeddings = {
        id(table.weight): table.weight for table in model.side_embeddings if table.weight.requires_grad
    }
    return {
        "config": asdict(config),
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_elements": sum(parameter.numel() for parameter in trainable.values()),
        "embedding_table_tensors": len(embeddings),
        "side_embedding_elements": sum(parameter.numel() for parameter in side_embeddings.values()),
    }
