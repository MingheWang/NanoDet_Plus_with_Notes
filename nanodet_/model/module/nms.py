import torch
from torchvision.ops import nms


def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)                            shape=(N,4)
        multi_scores (Tensor): shape (n, #class), where the last column                 shape=(N,81)
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.                                  100
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1          # 80
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)         # shape = (N,80,4)      扩展的原因是，一个特征点可能有多个类别
    scores = multi_scores[:, :-1]           # shape = (N,80)

    # filter out boxes with low scores
    valid_mask = scores > score_thr         # shape = (N,80)

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)                                                                           # shape = (M,4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)                                        # shape = (M,)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]                                       # shape = (M,)

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels

    # dets.shape = (K,5)        5 = x1, y1, x2, y2, score
    # keep: shape=(K,)          索引值
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:             # 100
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    为了对每个类独立执行 NMS，我们为所有的框添加了一个偏移量。
    偏移量仅依赖于类 idx，而且偏移量足够大，以至于来自不同类的框不会重叠。
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()                                                   # dict(type="nms", iou_threshold=0.6)
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()                       # 每个预测框x1y1x2y2中最大值   shape=(N,)
        offsets = idxs.to(boxes) * (max_coordinate + 1)         # 为了对每个类独立执行 NMS，我们为所有的框添加了一个偏移量。
                                                                # 偏移量仅依赖于类 idx，而且偏移量足够大，以至于来自不同类的框不会重叠。
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)           # 每张图片nms操作前预测框的最大数目
    # 如果该图中的预测框数目小于最大值
    if len(boxes_for_nms) < split_thr:  
        keep = nms(boxes_for_nms, scores, **nms_cfg_)      # 使用torchvision中的nms函数 ， 得到索引值 (M,)
        boxes = boxes[keep]
        scores = scores[keep]
    # 如果该图中的预测框数目大于最大值
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)      # (N,)  挑选出该图中的预测框，正确的预测结果为True ， 错误的为False
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)            # 筛选出该类别id的所有目标框的 索引     (ni,)
            keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)       # 对图中该类别的所有预测框运用NMS极大值抑制，选出该类别目标框NMS后的索引
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)          # NMS后的索引值 (M,)
        keep = keep[scores[keep].argsort(descending=True)]          # 再根据预测概率大小对索引值进行排序 (M,)   
        boxes = boxes[keep]
        scores = scores[keep]

    # 返回:
    #   boxes：shape=(M,5)    5 = x1, y1, x2, y2, score
    #   索引keep: shape=(M,)
    return torch.cat([boxes, scores[:, None]], -1), keep
