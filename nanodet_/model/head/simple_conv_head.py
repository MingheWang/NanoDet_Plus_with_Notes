import torch
import torch.nn as nn

from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from ..module.scale import Scale


class SimpleConvHead(nn.Module):
    def __init__(
        self,
        num_classes,                    # 80
        input_channel,                  # 192
        feat_channels=256,              # 192
        stacked_convs=4,
        strides=[8, 16, 32],            # [8, 16, 32, 64]
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        activation="LeakyReLU",
        reg_max=16,                     # 7
        **kwargs
    ):
        super(SimpleConvHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.reg_max = reg_max

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):     # 4
            chn = self.in_channels if i == 0 else self.feat_channels
            # 普通的nn.Conv2d卷积,groups=1  , ("conv", "norm", "act")
            # 192 --> 192
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,                 # 192
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
            # 普通的nn.Conv2d卷积,groups=1  , ("conv", "norm", "act")
            # 192 --> 192
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,                 # 192
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
        # 192 --> 80
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        # 192 --> 4*(7+1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1
        )
        # 每个检测头都共享相同的分类和回归的卷积网络，
        # 而分类和回归的最后一层的输出还需要乘以一个尺度参数self.scales
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

    def forward(self, feats):       # [c=192,c=192,c=192,c=192,c=192]
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            # 每个检测头共享相同的分类器self.cls_convs和回归器self.reg_convs
            for cls_conv in self.cls_convs:             # 192 --> 192 --> 192 --> 192 --> 192
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:             # 192 --> 192 --> 192 --> 192 --> 192
                reg_feat = reg_conv(reg_feat)
            cls_score = self.gfl_cls(cls_feat)                          # 192 --> 80
            bbox_pred = scale(self.gfl_reg(reg_feat)).float()           # 192 --> 32
            # output.shape = (B,80+32,Hi,Wi)
            output = torch.cat([cls_score, bbox_pred], dim=1)
            # outputs每个元素为output.shape=(B,112,Hi*Wi)
            outputs.append(output.flatten(start_dim=2))
        # outputs.shape:    [(B,112,Hi*Wi), (B,112,Hi*Wi), ...] --> (B,112,N1+N2+N3+N4) --> (B,N1+N2+N3+N4,112)
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        # 返回：outputs.shape=(B,N1+N2+N3+N4,112)
        return outputs
