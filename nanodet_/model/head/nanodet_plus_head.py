import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,                          # 96
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],                    # [8, 16, 32, 64]
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(NanoDetPlusHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.norm_cfg = norm_cfg

        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)      # loss_weight=2.0
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()                        
        for _ in self.strides:      # strides=[8, 16, 32, 64] 共有4个检测头，每个检测头都有两个深度可分离卷积组成
            cls_convs = self._buid_not_shared_head()            # 96 --> 96 --> 96
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList(
            [   # 96 --> 112 = 80 + 4*(7 + 1)
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1),
                    1,
                    padding=0,
                )
                for _ in self.strides       # strides=[8, 16, 32, 64]
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()             # 96 --> 96 --> 96
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # ConvModule_1的顺序order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act")      96 ---> 96
            # ConvModule_2的顺序order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act")      96 ---> 96
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:          # 96 --> 96 --> 96  
                feat = conv(feat)          
            output = gfl_cls(feat)          # 96 --> 112            (B,C,H,W)
            outputs.append(output.flatten(start_dim=2))             # (B,C,H*W)
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)        # (B,C,N1+N2+N3+N4) ---> (B,N1+N2+N3+N4,C)
        return outputs

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.                                          # shape=(B,N1+N2+N3+N4,C=112)
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.      # shape=(B,N1+N2+N3+N4,C=112)

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        device = preds.device
        batch_size = preds.shape[0]
        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        # 列表的每个元素对应：每个特征图上每个点在源图上的坐标，及步长   # (B,h*w,4)      4 = cx, cy, stride, stride
        # [(B,h*w,4), (B,h*w,4), (B,h*w,4), (B,h*w,4)]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        # 所有特征图上的特征点的坐标：shape = (B,N1+N2+N3+N4,4)     4 = cx, cy, stride, stride
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        # shape=(B,N1+N2+N3+N4,80)  分类
        # shape=(B,N1+N2+N3+N4,32)  回归
        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        # self.distribution_project(reg_preds)返回(B,N,4)，取值区间0~7
        # 乘以每个点处的步长：(B,N,4) * (B,N,1) ---> (B,N,4)    点到4条边的距离
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        # 计算处个每点特征点预测的目标框的x1y1x2y2
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # use auxiliary head to assign  使用辅助检测头用于标签分配
            # shape=(B,N1+N2+N3+N4,80)  
            # shape=(B,N1+N2+N3+N4,32)  
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            # self.distribution_project(reg_preds)返回(B,N,4)，取值区间0~7
            # 乘以每个点处的步长：(B,N,4) * (B,N,1) ---> (B,N,4)    点到4条边的距离
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            # 计算处个每点特征点预测的目标框(B,N,4), 4 = x1, y1, x2, y2
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            ##############################################################################
            # 每张图片对应一个(labels,label_scores,bbox_targets,dist_targets,num_pos_per_img)
            #                labels,                 # shape=(N,)  正样本对应相应目标框的类别id, 负样本对应类别id=80
            #                label_scores,           # shape=(N,)  正样本对应最佳的匹配iou值，负样本对应的为 0
            #                bbox_targets,           # shape=(N,4) 正样本对应目标框的xyxy坐标，负样本对应坐标全为0
            #                dist_targets,           # shape=(N,4)  正样本到对应的目标框的4条边的距离(还需除以步长s)， 负样本对应的值为0
            #                num_pos_per_img,        # 正样本数目
            # 通过map(list,zip(*))后：返回一个列表 ， 每个元素如下
            #                labels:    [(N,), (N,), ...]           正样本对应相应目标框的类别id, 负样本对应类别id=80
            #                label_scores:   [(N,), (N,), ...]      正样本对应最佳的匹配iou值，负样本对应的为 0
            #                bbox_targets:  [(N,4),(N,4),...]       正样本对应目标框的xyxy坐标，负样本对应坐标全为0
            #                dist_targets:  [(N,4),(N,4),...]       正样本到对应的目标框的4条边的距离(还需除以步长s)， 负样本对应的值为0
            #                num_pos_per_img:   []                  正样本数目
            ##############################################################################
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),                     # shape=(B,N1+N2+N3+N4,80) 
                center_priors,                              # shape = (B,N1+N2+N3+N4,4)     4 = cx, cy, stride, stride
                aux_decoded_bboxes.detach(),                # 计算处个每点特征点预测的目标框(B,N,4), 4 = x1, y1, x2, y2
                gt_bboxes,
                gt_labels,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        # 总损失loss , 每部分的损失loss_states = {质量分类损失, 预测框giou损失, 回归分布损失}
        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res
        )

        if aux_preds is not None:
            # 总损失aux_loss , 每部分的损失aux_loss_states = {质量分类损失, 预测框giou损失, 回归分布损失}
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        device = cls_preds.device
        # labels:    [(N,), (N,), ...]           正样本对应相应目标框的类别id, 负样本对应类别id=80
        # label_scores:   [(N,), (N,), ...]      正样本对应最佳的匹配iou值，负样本对应的为 0
        # bbox_targets:  [(N,4),(N,4),...]       正样本对应目标框的xyxy坐标，负样本对应坐标全为0
        # dist_targets:  [(N,4),(N,4),...]       正样本到对应的目标框的4条边的距离(已经除以步长s)， 负样本对应的值为0
        # num_pos_per_img:   []                  正样本数目
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )
        # shape=(B*N,) 将一个batch的值concat
        labels = torch.cat(labels, dim=0)
        # shape=(B*N,) 将一个batch的值concat
        label_scores = torch.cat(label_scores, dim=0)
        # shape=(B*N,4) 将一个batch的值concat
        bbox_targets = torch.cat(bbox_targets, dim=0)
        # shape=(B*N,80)    一个batch中所有的特征点的分类预测概率
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        # shape=(B*N,32)    一个batch中所有特征点的回归值
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        # shape=(B*N,4)     一个batch中所有特征点预测的回归坐标解码后的预测框坐标xyxy
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        #####################################################################################################
        # 计算分类损失，通过预测框与最佳匹配目标框的iou作为监督信号，计算预测框的分类损失   ---> 返回一个损失均值
        #####################################################################################################
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )
        # 所有的特征点中正样本的索引pos_inds.shape=(K,)
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]       # (K,80) --> (K,) 每个正样本的最大预测概率值
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            ###############################################################################################
            # 计算正样本的giou损失，通过计算预测框坐标与其对应的目标框坐标之间的iou损失  --->  返回一个损失均值
            ###############################################################################################
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],               # 正样本解码后的框的坐标  shape=(K,4)     4 = xyxy
                bbox_targets[pos_inds],                 # 正样本对应的目标框坐标  shape=(K,4)     4 = xyxy
                weight=weight_targets,                  # 每个正样本的分类预测最大值  shape=(K,)
                avg_factor=bbox_avg_factor,
            )
            # shape=(N,4)   正样本到对应的目标框的4条边的距离(还需除以步长s)， 负样本对应的值为0
            dist_targets = torch.cat(dist_targets, dim=0)
            ###############################################################################################
            # 计算正样本的预测回归框的分布损失  --->  返回一个损失均值
            ###############################################################################################
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),                  # (K,32) ---> (4*K,8)
                dist_targets[pos_inds].reshape(-1),                                 # (K,4) ---> (4*K,) 已经除过对应的步长s
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),           # (K,) ---> (K,1) ---> (K,4) ---> (4*K,)
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        # 总损失 = 质量分类损失 + 预测框giou损失 + 预测框回归分布损失
        loss = loss_qfl + loss_bbox + loss_dfl      
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:                                                                           # 对一张图进行处理          N = N1+N2+N3+N4
            cls_preds (Tensor): Classification predictions of one image,                # shape=(N,80) 
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with           # shape = (N,4)     4 = cx, cy, stride, stride
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,           # 计算处个每点特征点预测的目标框(N,4), 4 = x1, y1, x2, y2
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = center_priors.size(0)                          # N
        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)          
        gt_labels = torch.from_numpy(gt_labels).to(device)          
        num_gts = gt_labels.size(0)                                 # M
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)              # (M,4)

        bbox_targets = torch.zeros_like(center_priors)              # (N,4)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(                            # (N,)  默认为80
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)         # (N,)   默认为 0
        # No target
        if num_gts == 0:
            return labels, label_scores, bbox_targets, dist_targets, 0
        #################################################################################
        # assigned_gt_inds.shape=(N,)     若是特征点为前景则对应前景的序号  若是背景则对应0
        # assigned_labels.shape=(N,)      每个特征点对应的类别id  背景则为-1
        # max_overlaps.shape=(N,)         每个预测框对应的最佳匹配的iou值   背景则为 -INF
        #################################################################################
        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )
        ####################################################################
        # 返回：
        #   pos_inds:  所有特征点中的正样本的索引  shape=(K,1)
        #   neg_inds:  所有特征点中的负样本的索引  shape=(N-K,1)
        #   pos_assigned_gt_inds:   每个正样本对应的目标框的索引  shape=(K,)
        #   pos_gt_bboxes:  正样本对应的目标框坐标  shape=(K,4)
        #####################################################################
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            # shape=(N,4) 正样本对应目标框的xyxy坐标，负样本对应坐标全为0
            bbox_targets[pos_inds, :] = pos_gt_bboxes               
            # shape=(N,4)  正样本到对应的目标框的4条边的距离(还需除以步长s)， 负样本对应的值为0
            dist_targets[pos_inds, :] = (
                # 计算每个正样本的中心点到其对应的目标框4个边的距离,shape=(K,4)
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]          # 除以步长shape=(K,1)
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            # shape=(N,)  正样本对应相应目标框的类别id, 负样本对应类别id=80
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            # shape=(N,)  正样本对应最佳的匹配iou值，负样本对应的为 0
            label_scores[pos_inds] = pos_ious
        return (
            labels,                 # shape=(N,)  正样本对应相应目标框的类别id, 负样本对应类别id=80
            label_scores,           # shape=(N,)  正样本对应最佳的匹配iou值，负样本对应的为 0
            bbox_targets,           # shape=(N,4) 正样本对应目标框的xyxy坐标，负样本对应坐标全为0
            dist_targets,           # shape=(N,4)  正样本到对应的目标框的4条边的距离(已经除过步长s)， 负样本对应的值为0
            num_pos_per_img,        # 正样本数目
        )

    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        # 从所有的特征点中，获取正样本的位置索引 pos_inds.shape=(K,1)
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        # 从所有的特征点中，获取负样本的位置索引 neg_inds.shape=(N-K,1)
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1          # shape=(K,)

        if gt_bboxes.numel() == 0:      # 如果该图中没有目标
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]              # shape=(K,4)

        # 返回：
        #   pos_inds:  所有特征点中的正样本的索引  shape=(K,1)
        #   neg_inds:  所有特征点中的负样本的索引  shape=(N-K,1)
        #   pos_assigned_gt_inds:   每个正样本对应的目标框的索引  shape=(K,)
        #   pos_gt_bboxes:  正样本对应的目标框坐标  shape=(K,4)
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.          # shape=(B,N1+N2+N3+N4,C=112)
            meta (dict): Meta info.                     # 其中的图片"img"的shape=(B,3,H,W)      
        """
        # cls_scores = (B,N1+N2+N3+N4,80)
        # bbox_preds = (B,N1+N2+N3+N4,32)
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        # 返回：极大值抑制后的结果
        # result_list = [(bboxes,labels), (bboxes,labels), ...]         bboxes.shape=(K,5) , labels.shape=(K,)
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            # 一张图的所有预测结果：
            # det_restlt是一个字典：key为类别id . 值为每个类别的预测框[[x1,y1.x2,y2,score], [x1,y1.x2,y2,score], ...]
            det_result = {}
            # bboxes.shape=(K,5) , labels.shape=(K,)
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )                       # np.linalg.inv返回给定矩阵的逆矩阵
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                # 一张图的所有预测结果：
                # det_restlt是一个字典：key为类别id . 值为每个类别的预测框[[x1,y1.x2,y2,score], [x1,y1.x2,y2,score], ...]
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            # 一个batch中的所有预测结果：
            # det_results为一个字典：
            # key为图片id , 
            # value为一张图的预测字典det_result={"cls_id":[[x1,y1.x2,y2,score], [x1,y1.x2,y2,score], ...] , ...}
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        # 列表的每个元素对应：每个特征图上点在源图上的坐标，及步长   # (B,h*w,4)
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        # 所有特征图上的特征点在原图上的坐标：shape = (B,N1+N2+N3+N4,4)
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        # self.distribution_project(reg_preds)返回(B,N,4)，取值区间0~7
        # 乘以每个点处的步长：(B,N,4) * (B,N,1) ---> (B,N,4) 得到每点到四个边框的真实距离
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        # 计算出个每点特征点预测的目标框的x1y1x2y2
        # shape=(B,N,4)
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        # shape=(B,N,80)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]                  # shape=(N,80), shape=(N,4)
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)          # shape=(N,81)
            # 返回results = (bboxes, labels)    
            # bboxes.shape=(K,5) , labels.shape=(K,)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        # result_list = [(bboxes,labels), (bboxes,labels), ...]         bboxes.shape=(K,5) , labels.shape=(K,)
        return result_list

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)     # y, x.shape = (h,w)
        y = y.flatten()     # shape=(h*w,)      原图坐标点
        x = x.flatten()     # shape=(h*w,)
        strides = x.new_full((x.shape[0],), stride)     # shape=(h*w,)
        proiors = torch.stack([x, y, strides, strides], dim=-1)         # shape=(h*w,4)     4=x,y,s,s   特征图上每个点在原图上的坐标，及步长
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)            # shape=(1,h*w,4) ---> (B,h*w,4)

    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)
