import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weighted_loss


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert (
        len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()                   # shape=(B*N,80)
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    #############################################
    #   先计算负样本的损失 ---> shape=(B*N,80)
    #############################################
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    # 在所有的特征点中选择正样本的索引  shape=(K,)
    pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(1)
    # 挑选出正样本的标签  shape=(K,)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    # 正样本的预测的分类分值受到预测框的最佳匹配的目标框的(iou质量)监督     (K,) - (K,) --> (K,)
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    ###########################################################
    # 计算正样本的分类回归损失, 更新之前的损失 ---> shape=(B*N,80)
    ###########################################################
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)
    ################################################
    # 计算每一个特征点的所有类别损失 ---> shape=(B*N,) 
    ################################################
    loss = loss.sum(dim=1, keepdim=False)
    return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor):                                  # shape=(K*4,8)    正样本预测的到对应目标框4个边预测出的距离
            Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor):                                 # shape=(K*4,)     正样本预测的到对应目标框4个边的实际距离,float,已经除过对应的步长s
            Target distance label for bounding boxes with shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()       # 该标签值的左整数边界值
    dis_right = dis_left + 1      # 该标签值的右整数边界值
    weight_left = dis_right.float() - label                     # 该标签据其右整数边界的距离
    weight_right = label - dis_left.float()                     # 该标签据其左整数边界的距离
    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left                                 # one-hot ，只计算为1处的损失
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor):    一个batch中所有的特征点的分类预测概率   shape=(B*N,80)
                Predicted joint representation of 
                classification and quality (IoU) estimation with shape (N, C),      
                C is the number of classes.                                                 
            target (tuple([torch.Tensor])): 
                Target category label with shape (N,)           shape=(B*N,)    正样本对应相应目标框的类别id, 负样本对应类别id=80
                and target quality label with shape (N,).       shape=(B*N,)    正样本对应最佳的匹配目标框的iou值,负样本对应的为 0
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction            # "mean"
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor):                                                            shape=(K*4,8)
                Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor):                                                          shape=(K*4,)    已经除过对应的步长s
                Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional):                                                shape=(K*4,)    每个正样本预测到的最大分类概率值
                The weight of loss for each prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction            # default "mean"
        # 计算计算回归框的分布损失
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls


