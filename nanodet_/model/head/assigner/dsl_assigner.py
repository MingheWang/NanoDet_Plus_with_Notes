import torch
import torch.nn.functional as F

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class DynamicSoftLabelAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with
    dynamic soft label assignment.

    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matchs for each gt. Default 13.
        iou_factor (float): The scale factor of iou cost. Default 3.0.
    """

    def __init__(self, topk=13, iou_factor=3.0):
        self.topk = topk
        self.iou_factor = iou_factor

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
    ):
        """Assign gt to priors with dynamic soft label assignment.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]                            # (N,80)
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.                     # (N,4)
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.                         # (N,4)
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.                 # (M,4)
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].                                                       # (M,)

        Returns:
            :obj:`AssignResult`: The assigned result.
                    assigned_gt_inds.shape=(N,)     若是特征点为前景则对应前景的序号  若是背景则对应0
                    assigned_labels.shape=(N,)      每个特征点对应的类别id  背景则为-1
                    max_overlaps.shape=(N,)         每个预测框对应的最佳匹配的iou值   背景则为 -INF
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)              # M
        num_bboxes = decoded_bboxes.size(0)     # N

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)      # (N,)  默认为 0 , 全为背景

        prior_center = priors[:, :2]    # (N,2) 2 = cx, cy
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]          # (N,1,2) - (M,2) ---> (N,M,2)  每个真实的目标框左上角相对于每个特征点中的位置
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]          # (M,2) - (N,1,2) ---> (N,M,2)  每个真实的目标框右下角相对于每个特征点中的位置

        deltas = torch.cat([lt_, rb_], dim=-1)                  # (N,M,4)
        is_in_gts = deltas.min(dim=-1).values > 0               # 只有当4个值都大于0的时候，特征点才会在目标框的内部    (N,M)   True or False
        valid_mask = is_in_gts.sum(dim=1) > 0                   # (N,)  只有当特征点在任意一个目标框中，valid_mask为True

        ##########################################################
        # 从所有的特征点中筛选出处在目标框中的特征点的预测值box和score
        ##########################################################
        valid_decoded_bbox = decoded_bboxes[valid_mask]         # (T,4)
        valid_pred_scores = pred_scores[valid_mask]             # (T,80)
        num_valid = valid_decoded_bbox.size(0)                  # T

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0         # 该图中没有目标，所有的点全为背景
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1, dtype=torch.long
                )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)                # giou:     shape=(T,M)
        ##########################
        # 计算IOU损失   shape=(T,M)
        ##########################
        iou_cost = -torch.log(pairwise_ious + 1e-7)
        # shape:  (M,80) ---> (1,M,80) ---> (T,M,80)
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        # shape:  (T,80) ---> (T,1,80) ---> (T,M,80)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        # (T,M,80)*(T,M,1) ---> (T,M,80)
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores
        #############################
        # 计算分类损失  shape=(T,M,80)
        #############################
        cls_cost = F.binary_cross_entropy(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)
        # shape=(T,M)
        cls_cost = cls_cost.sum(dim=-1)     
        # shape=(T,M)
        cost_matrix = cls_cost + iou_cost * self.iou_factor
        #########################################################################################
        # matched_gt_inds：对选出正预测样本，获得其对应的目标框序号。   shape=(K,)
        # matched_pred_ious: 对应的iou值。  shape=(K,)
        # valid_mask: 在模型输出的所有特征点中，只保留以上筛选后的检测框的位置值为True。  shape=(N,)
        #########################################################################################
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )
        #####################################################################
        # convert to AssignResult format
        # 将以上得到的目标框序号加入到全局的assigned_gt_inds范围上。  shape=(N,)    
        # 若是背景则对应0，若是前景则对应前景的序号
        #####################################################################
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        # shape=(N,)    每个特征点对应的类别id  背景则为-1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)              
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()   
        # shape=(N,)    每个预测框对应的最佳匹配的iou值   背景则为 -INF       
        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Use sum of topk pred iou as dynamic k. Refer from OTA and YOLOX.

        Args:
            cost (Tensor): Cost matrix.                             # shape=(T,M)
            pairwise_ious (Tensor): Pairwise iou matrix.            # shape=(T,M)
            num_gt (int): Number of gt.                             # shape=(M,)
            valid_mask (Tensor): Mask for valid bboxes.             # shape=(N,)
        """
        #------------------------------------------------------------------------------------------------------------#
        #   1\根据iou值，每个目标框选择topk个预测框的iou   shape=(13,M)
        #   2\共M个目标框，每个目标框根据前topk个iou之和，动态地选择与该目标框的匹配损失最小的前p个预测框，匹配损失小的，置为1
        #   3\因为一个预测框可能处在多个目标框中，所以在这里选择匹配损失最小的匹配作为该预测框的匹配
        #   4\完成以上的处理后，对选出正预测样本，获得其对应的目标框序号以及对应的iou
        #------------------------------------------------------------------------------------------------------------#
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)             # 根据iou值，每个目标框选择topk个预测框的iou   shape=(13,M)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)     # shape=(M,)
        for gt_idx in range(num_gt):            # 共M个目标框，每个目标框根据前topk个iou之和，动态地选择与该目标框的匹配损失最小的前p个预测框
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0       # 匹配损失小的，置为1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1         # shape=(T,)   如果某个预测被目标框匹配到，则该值为True
        if prior_match_gt_mask.sum() > 0:   # 该图存在被匹配的预测框
            ################################################################################
            # 因为一个预测框可能处在多个目标框中，所以在这里选择匹配损失最小的匹配作为该预测框的匹配
            ################################################################################
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)      # 选择每一个与预测框匹配损失最小的目标框
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0     # 该预测框选择最小匹配损失的匹配作为该匹配
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        ############################################################################################################
        # 使用 mask1[mask1.clone()] = mask2 ， 其中 mask2 只是在 mask1 中的真值位置上(而不是在所有的位置)进行进一步筛选的。
        # 使得 mask2 在 mask1 上的对应位置上进一步更改。这样就保留了mask1中的顺序
        ############################################################################################################
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        # 完成以上的处理后，对选出正预测样本，获得其对应的目标框序号
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        # 以及对应的iou
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
