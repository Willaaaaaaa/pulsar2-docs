import argparse
from pathlib import Path

from typing import List

import cv2
import torch
from pulsar2_run_helper.utils import Logger, sanitize

LOGGER = Logger(colors=False)


def get_parser():
    parser = argparse.ArgumentParser("CLI tools for pre-processing and post-processing.", add_help=True)

    parser.add_argument(
        "--image_path",
        type=str,
        help="The path of image file.",
    )
    parser.add_argument(
        "--axmodel_path",
        type=str,
        required=True,
        help="The path of compiled axmodel.",
    )
    parser.add_argument(
        "--intermediate_path",
        type=str,
        required=True,
        help="The path of intermediate data bin.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path of output files.",
    )
    parser.add_argument(
        "--letterbox_size",
        type=int,
        default=640,
        help="Image size for croping (default: 640).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of classes (default: 80).",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.45,
        help="Threshold of score (default: 0.45).",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.45,
        help="Threshold of NMS (default: 0.45).",
    )
    parser.add_argument("--pre_processing", action="store_true", help="Do pre processing.")
    parser.add_argument("--post_processing", action="store_true", help="Do post processing.")

    return parser


def pre_processing(args):
    from pulsar2_run_helper.preprocessing import get_input_info, preprocess_detection

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Not found image file at '{image_path}'")

    axmodel_path = Path(args.axmodel_path)
    if not axmodel_path.exists():
        raise FileNotFoundError(f"Not found compiled axmodel at '{axmodel_path}'")

    img_transformed, _ = preprocess_detection(
        str(image_path),
        letterbox_size=args.letterbox_size,
        color=(0, 0, 0),
        enable_bgr_to_rgb=True,
    )
    input_names = get_input_info(axmodel_path)
    if len(input_names) != 1:
        raise NotImplementedError(f"Currently only supports length 1, but got {input_names}")

    intermediate_path = Path(args.intermediate_path)
    if not intermediate_path.exists():
        raise FileNotFoundError(f"Not found directory at '{intermediate_path}'")

    output_path = intermediate_path / f"{sanitize(input_names[0])}.bin"
    output_path.write_bytes(img_transformed.tobytes())
    LOGGER.info(f"Write [{input_names[0]}] to '{output_path}' successfully.")


def post_processing(args):

    from pulsar2_run_helper.postprocessing import get_yolov5_predictions

    axmodel_path = Path(args.axmodel_path)
    if not axmodel_path.exists():
        raise FileNotFoundError(f"Not found compiled axmodel at '{axmodel_path}'")

    intermediate_path = Path(args.intermediate_path)
    if not intermediate_path.exists():
        raise FileNotFoundError(f"Not found directory at '{intermediate_path}'")

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Not found image file at '{image_path}'")

    origin_image_shapes = _prepare_coord_scale_shapes([args.image_path])

    detection_outputs = get_yolov5_predictions(
        model_path=args.axmodel_path,
        intermediate_path=args.intermediate_path,
        origin_image_shapes=origin_image_shapes,
        num_classes=args.num_classes,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
    )

    for det_output in detection_outputs:
        num_objects = len(det_output["labels"])
        LOGGER.info(f"Number of detected objects: {num_objects}")
        for label, score, box in zip(det_output["labels"], det_output["scores"], det_output["boxes"]):
            box = torch.round(box).int().tolist()
            LOGGER.info(f"{label:2d}: {score*100:.02f}%, [{box[0]:3d}, {box[1]:3d}, {box[2]:3d}, {box[3]:3d}]")


def _prepare_coord_scale_shapes(image_paths: List[str]):

    image_origin_shapes = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_origin_shapes.append(image.shape[:-1])

    return image_origin_shapes


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    LOGGER.debug(f"Command Line Args: {args}")

    if args.pre_processing:
        pre_processing(args)
    if args.post_processing:
        post_processing(args)


if __name__ == "__main__":
    cli_main()
