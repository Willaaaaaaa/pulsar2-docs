# Copyright (c) 2020, yolort team. All rights reserved.

from typing import Dict, List

import torch
from torch import nn, Tensor
from torchvision.ops import box_convert, boxes as box_ops

from .det_utils import decode_single


class YOLOHead(nn.Module):
    """
    A regression and classification head for use in YOLO.

    Args:
        in_channels (List[int]): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        strides (List[int]): number of strides of the anchors
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels: List[int], num_anchors: int, strides: List[int], num_classes: int):

        super().__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels] * len(strides)
        self.num_anchors = num_anchors  # anchors
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.strides = strides

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        all_pred_logits = []  # inference output

        for pred_logit in x:
            # Permute output from (N, A * K, H, W) to (N, A, H, W, K)
            N, _, H, W = pred_logit.shape
            pred_logit = pred_logit.view(N, self.num_anchors, -1, H, W)
            # Size=(N, A, H, W, K)
            pred_logit = pred_logit.permute(0, 1, 3, 4, 2).contiguous()

            all_pred_logits.append(pred_logit)

        return all_pred_logits


def _concat_pred_logits(
    head_outputs: List[Tensor],
    grids: List[Tensor],
    shifts: List[Tensor],
    strides: Tensor,
) -> Tensor:
    # Concat all pred logits
    batch_size, _, _, _, K = head_outputs[0].shape

    # Decode bounding box with the shifts and grids
    all_pred_logits = []

    for head_output, grid, shift, stride in zip(head_outputs, grids, shifts, strides):
        head_feature = torch.sigmoid(head_output)
        pred_xy, pred_wh = decode_single(head_feature[..., :4], grid, shift, stride)
        pred_logits = torch.cat((pred_xy, pred_wh, head_feature[..., 4:]), dim=-1)
        all_pred_logits.append(pred_logits.view(batch_size, -1, K))

    all_pred_logits = torch.cat(all_pred_logits, dim=1)

    return all_pred_logits


def _decode_pred_logits(pred_logits: Tensor):
    """
    Decode the prediction logit from the PostPrecess.
    """
    # Compute conf
    # box_conf x class_conf, w/ shape: num_anchors x num_classes
    scores = pred_logits[..., 5:] * pred_logits[..., 4:5]
    boxes = box_convert(pred_logits[..., :4], in_fmt="cxcywh", out_fmt="xyxy")

    return boxes, scores


class PostProcess(nn.Module):
    """
    Performs Non-Maximum Suppression (NMS) on inference results

    Args:
        strides (List[int]): Strides of the AnchorGenerator.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
    """

    def __init__(
        self,
        strides: List[int],
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
    ) -> None:

        super().__init__()
        self.strides = strides
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(
        self,
        head_outputs: List[Tensor],
        grids: List[Tensor],
        shifts: List[Tensor],
    ) -> List[Dict[str, Tensor]]:
        """
        Perform the computation. At test time, postprocess_detections is the final layer of YOLO.
        Decode location preds, apply non-maximum suppression to location predictions based on conf
        scores and threshold to a detections_per_img number of output predictions for both confidence
        score and locations.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            grids (List[Tensor]): Anchor grids.
            shifts (List[Tensor]): Anchor shifts.
        """

        batch_size = head_outputs[0].shape[0]
        device = head_outputs[0].device
        dtype = head_outputs[0].dtype
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device).to(dtype=dtype)

        all_pred_logits = _concat_pred_logits(head_outputs, grids, shifts, strides)
        detections: List[Dict[str, Tensor]] = []

        for idx in range(batch_size):  # image idx, image inference
            pred_logits = all_pred_logits[idx]
            boxes, scores = _decode_pred_logits(pred_logits)
            # remove low scoring boxes
            inds, labels = torch.where(scores > self.score_thresh)
            boxes, scores = boxes[inds], scores[inds, labels]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring head_outputs
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            detections.append({"scores": scores, "labels": labels, "boxes": boxes})

        return detections
