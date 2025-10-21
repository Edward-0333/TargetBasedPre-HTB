# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData
from models import MapEncoder
from models import LinearScorerLayer
from torch.nn.utils.rnn import pad_sequence


class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.map_encoder = MapEncoder(
            dim=64,
            polygon_channel=4,
            use_lane_boundary=True,
        )
        self.linear_scorer_layer = LinearScorerLayer(T=30, d=256)

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None
        test = data.batch.cpu().numpy()
        test2 = data.agent_lane_id_target.cpu().numpy()
        test3 = data.y.cpu().numpy()
        test4 = data.padding_mask.cpu().numpy()
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        batch = data.batch
        batch_size = int(data.num_graphs)
        if batch_size == 0:
            global_embed_batch = global_embed.new_zeros((0, 0, global_embed.size(-1)))
            padding_mask_batch = data['padding_mask'].new_zeros((0, 0))
        else:
            split_sizes = torch.bincount(batch, minlength=batch_size)
            if split_sizes.device.type != 'cpu':
                split_sizes = split_sizes.cpu()
            split_sizes = split_sizes.tolist()
            global_embed_batch = global_embed.split(split_sizes, dim=0)
            padding_mask_batch = data['padding_mask'].split(split_sizes, dim=0)
            global_embed_batch = pad_sequence(global_embed_batch, batch_first=True, padding_value=0.0)
            padding_mask_batch = pad_sequence(padding_mask_batch, batch_first=True, padding_value=True)
        lane_features, valid_mask = self.map_encoder(data=data)
        map_key_padding = ~valid_mask.any(-1)

        logits, probs = self.linear_scorer_layer(
            global_embed_batch,
            lane_features,
            agent_mask=padding_mask_batch[:,:,self.historical_steps:],
            lane_mask=map_key_padding,
        )

        # y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return logits, probs

    def training_step(self, data, batch_idx):
        logits, probs = self(data)
        target_lane_id = data['agent_lane_id_target'][:, self.historical_steps:].long()
        batch = data.batch
        batch_size = int(data.num_graphs)
        split_sizes = torch.bincount(batch, minlength=batch_size)
        if split_sizes.device.type != 'cpu':
            split_sizes = split_sizes.cpu()
        split_sizes = split_sizes.tolist()
        target_lane_id = target_lane_id.split(split_sizes, dim=0)
        target_lane_id = pad_sequence(target_lane_id, batch_first=True, padding_value=-1)
        ignore_index = -100
        target_lane_id = target_lane_id.masked_fill(target_lane_id == -1, ignore_index)

        agent_mask = data['padding_mask'][:, self.historical_steps:]
        agent_mask = agent_mask.split(split_sizes, dim=0)
        agent_mask = pad_sequence(agent_mask, batch_first=True, padding_value=True)
        B, N, T, K = probs.shape
        loss = F.cross_entropy(
            logits.view(-1, K),
            target_lane_id.view(-1),
            reduction='none',
            ignore_index=ignore_index
        ).view(B, N, T)
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        if agent_mask is not None:
            valid = valid * (~agent_mask).float()
        loss = (loss * valid).sum() / valid.sum()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        logits, probs = self(data)
        target_lane_id = data['agent_lane_id_target'][:, self.historical_steps:].long()
        batch = data.batch
        batch_size = int(data.num_graphs)
        split_sizes = torch.bincount(batch, minlength=batch_size)
        if split_sizes.device.type != 'cpu':
            split_sizes = split_sizes.cpu()
        split_sizes = split_sizes.tolist()
        target_lane_id = target_lane_id.split(split_sizes, dim=0)
        target_lane_id = pad_sequence(target_lane_id, batch_first=True, padding_value=-1)
        ignore_index = -100
        target_lane_id = target_lane_id.masked_fill(target_lane_id == -1, ignore_index)

        agent_mask = data['padding_mask'][:, self.historical_steps:]
        agent_mask = agent_mask.split(split_sizes, dim=0)
        agent_mask = pad_sequence(agent_mask, batch_first=True, padding_value=True)
        B, N, T, K = probs.shape
        loss = F.cross_entropy(
            logits.view(-1, K),
            target_lane_id.view(-1),
            reduction='none',
            ignore_index=ignore_index
        ).view(B, N, T)
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        if agent_mask is not None:
            valid = valid * (~agent_mask).float()
        loss = (loss * valid).sum() / valid.sum()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 每个 epoch 调度一次
                "frequency": 1,  # 调度频率
                "name": "cosine_annealing",
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
