from typing import Dict, List, Optional, Tuple

import cv2

import numpy as np

from pulsar2_run_helper.yolort import AnchorGenerator, PostProcess, YOLOHead

from torch import nn, Tensor

_GROW_WIDTHS = [256, 512, 1024]

STRIDES = [8, 16, 32]

ANCHOR_GRIDS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]


class LetterboxPreProcessing:
    def __init__(
        self,
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scaleup: bool = True,
        stride: int = 32,
    ) -> None:

        self.color = color
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, image: np.ndarray, new_shape: Tuple[int, ...] = (640, 640)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return image, ratio, (dw, dh)


class YOLOv5PostProcessing(nn.Module):
    """
    Implements YOLO series model.

    The input to the model is expected to be a batched tensors, of shape ``[N, C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    """

    def __init__(
        self,
        depth_multiple: float,
        width_multiple: float,
        num_classes: int,
        # Anchor parameters
        strides: Optional[List[int]] = None,
        anchor_grids: Optional[List[List[float]]] = None,
        anchor_generator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        # Post Process parameter
        score_thresh: float = 0.005,
        nms_thresh: float = 0.45,
        detections_per_img: int = 300,
        post_process: Optional[nn.Module] = None,
    ):
        super().__init__()
        out_channels = [int(gw * width_multiple) for gw in _GROW_WIDTHS]

        if strides is None:
            strides: List[int] = STRIDES

        if anchor_grids is None:
            anchor_grids: List[List[float]] = ANCHOR_GRIDS

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(strides, anchor_grids)
        self.anchor_generator = anchor_generator

        if head is None:
            head = YOLOHead(
                out_channels,
                anchor_generator.num_anchors,
                anchor_generator.strides,
                num_classes,
            )
        self.head = head

        if post_process is None:
            post_process = PostProcess(
                anchor_generator.strides,
                score_thresh,
                nms_thresh,
                detections_per_img,
            )
        self.post_process = post_process

    def forward(self, features: List[Tensor]) -> List[Dict[str, Tensor]]:
        """
        Args:
            features (List[Tensor]): Expects a Tensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]

        Returns:
            result (list[Dict[Tensor]]): the output from the model.
                During testing, it returns List[Dict[Tensor]] contains additional fields
                like `scores`, `labels` and `boxes`.
        """
        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        grids, shifts = self.anchor_generator(features)

        # compute the detections
        detections = self.post_process(head_outputs, grids, shifts)

        return detections
