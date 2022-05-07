# Copyright 2021 RangiLyu.
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

import copy

import torch

from ..head import build_head
from .one_stage_detector import OneStageDetector


class NanoDetPlus(OneStageDetector):
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,                     # 10
    ):
        super(NanoDetPlus, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        # 当self.epoch大于10时，将输入到aux_fpn网络中的feat从计算图中剥离，将原网络fpn的输出fpn_feat从计算图中剥离。
        # 将fpn和aux_fpn的对应输出的特征图在通道维度上进行concat,使得通道数增加到原来的2倍，2*96=192
        if self.epoch >= self.detach_epoch:
            # detach将backbone输出的三个特征图从当前的计算图中剥离出来，得到的新的tensor永远不需要计算其梯度，不具有grad
            # 因为self.aux_fpn = copy.deepcopy(self.fpn) ， 所以self.aux_fpn和self.fpn结构相同
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            # 将fpn和aux_fpn的对应输出的特征图在通道维度上进行concat,使得通道数增加到原来的2倍，2*96=192
            dual_fpn_feat = (
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        # 当self.epoch小于于10时，同时训练aux_fpn和fpn两个网络，包括backbone
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            # 将fpn和aux_fpn的对应输出的特征图在通道维度上进行concat,使得通道数增加到原来的2倍，2*96=192
            dual_fpn_feat = (
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        
        # 将fpn网络输出(chennal=96)输入到原网络的self.head中。
        head_out = self.head(fpn_feat)      # ---> (B,N1+N2+N3+N4,C=112)

        # 将aux_fpn和fpn网络的输出concat后的输出(chennal=192)输入到self.aux_head网络中。
        aux_head_out = self.aux_head(dual_fpn_feat)         # ---> (B,N1+N2+N3+N4,C=112)
        # 如果既有大网络头又有小网络头，
        # 则loss为大网络的总损失加上小网络的总损失，否者只有小网络的总损失
        # 如果既有大网络头又有小网络头，
        # 则loss_states返回“大网络头”的各项损失，否者只返回小网络头的各项损失
        loss, loss_states = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        return head_out, loss, loss_states
