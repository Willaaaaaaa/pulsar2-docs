from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch

from pulsar2_run_helper.utils import get_tensor_value_info, sanitize

from torch import Tensor


def get_max_indices(model_path: str, intermediate_path: str, topk: int = 5):
    output_data = parse_output_data(model_path, intermediate_path)
    output_data = list(output_data.values())[0]
    max_indices = np.argsort(output_data, axis=1)[:, ::-1][:, :topk][0]
    scores = [output_data[0][i] for i in max_indices]

    return max_indices, scores


@torch.inference_mode()
def get_yolov5_predictions(
    model_path: str,
    intermediate_path: str,
    origin_image_shapes: List[Tuple[int, ...]],
    num_classes: int = 80,
    score_thresh: float = 0.45,
    nms_thresh: float = 0.45,
):

    from pulsar2_run_helper.pipeline import YOLOv5PostProcessing
    from pulsar2_run_helper.yolort.det_utils import scale_coords

    all_head_features = get_yolov5_heads(model_path, intermediate_path)

    yolov5_postprocess = YOLOv5PostProcessing(
        depth_multiple=0.33,
        width_multiple=0.5,
        num_classes=num_classes,
        # Post Process parameter
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=300,
    )
    yolov5_postprocess.eval()

    detection_outputs = yolov5_postprocess(all_head_features)

    for origin_img_shape, det_output in zip(origin_image_shapes, detection_outputs):
        scale_coords((640, 640), det_output["boxes"], origin_img_shape)

    return detection_outputs


def get_yolov5_heads(model_path: str, output_path: str) -> List[Tensor]:
    output_data = parse_output_data(model_path, output_path)

    all_head_features = []
    for _, head_feature in output_data.items():
        head_feature = torch.from_numpy(head_feature)
        head_feature = head_feature.permute(0, 3, 1, 2)  # NHWC -> NCHW
        all_head_features.append(head_feature)

    return all_head_features


def parse_output_data(model_path: str, output_path: str) -> Dict[str, np.ndarray]:
    output_info = get_output_info(model_path)

    output_data: Dict[str, np.ndarray] = {}
    for k, v in output_info.items():
        data_path = Path(f"{output_path}/{sanitize(k)}.bin")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Could not find the expected key '{k}', please double check your pulsar run output directory.",
            )
        data = data_path.read_bytes()
        output_data[k] = np.frombuffer(data, dtype=v["tensor_type"]).reshape(v["shape"]).copy()

    return output_data


def get_output_info(model_path: str):
    """
    Returns the shape and tensor type of all outputs.
    """

    model_obj = onnx.load(model_path)
    model_graph = model_obj.graph

    output_info = {}
    for tensor_info in model_graph.output:
        output_info.update({tensor_info.name: get_tensor_value_info(tensor_info)})
    return output_info
