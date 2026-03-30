# Copyright (c) 2020, yolort team. All rights reserved.

from typing import Tuple

import torch
from torch import Tensor


def encode_single(reference_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Encode a set of anchors with respect to some reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        anchors_tuple (Tensor): boxes to be encoded
    """
    reference_boxes = torch.sigmoid(reference_boxes)

    pred_xy = reference_boxes[:, :2] * 2.0 - 0.5
    pred_wh = (reference_boxes[:, 2:4] * 2) ** 2 * anchors
    pred_boxes = torch.cat((pred_xy, pred_wh), 1)

    return pred_boxes


def decode_single(
    rel_codes: Tensor,
    grid: Tensor,
    shift: Tensor,
    stride: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Args:
        rel_codes (Tensor): Encoded boxes
        grid (Tensor): Anchor grids
        shift (Tensor): Anchor shifts
        stride (int): Stride
    """
    pred_xy = (rel_codes[..., 0:2] * 2.0 - 0.5 + grid) * stride
    pred_wh = (rel_codes[..., 2:4] * 2.0) ** 2 * shift

    return pred_xy, pred_wh


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """

    if ratio_pad is None:  # calculate from img0_shape
        # gain  = old / new
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        # wh padding
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """

    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
