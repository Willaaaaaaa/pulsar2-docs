from typing import List, Tuple

import cv2
import onnx


def get_input_info(model_path: str) -> List[str]:
    """
    Returns the names list of axmodel.
    """

    model_obj = onnx.load(model_path)

    if hasattr(model_obj, "graph"):
        inits = set(_.name for _ in model_obj.graph.initializer)
        return [_.name for _ in model_obj.graph.input if _.name not in inits]
    return list(model_obj.input)


def preprocess_classification(
    image_path: str,
    crop_size: int = 224,
    resize_size: int = 256,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    add_tophat: bool = True,
):

    from pulsar2_run_helper.pipeline import LetterboxCenterCropPreProcessing

    img = cv2.imread(image_path)

    # Pre-Processing
    preprocess = LetterboxCenterCropPreProcessing(
        crop_size,
        resize_size=resize_size,
        mean=mean,
        std=std,
        add_tophat=add_tophat,
    )

    img_transformed = preprocess(img)  # Resize and Center crop
    img_transformed = img_transformed[None]  # HWC -> NHWC

    return img_transformed


def preprocess_detection(
    image_path: str,
    letterbox_size: Tuple[int, ...] = (640, 640),
    color: Tuple[int, ...] = (114, 114, 114),
    enable_bgr_to_rgb: bool = True,
):

    from pulsar2_run_helper.pipeline import LetterboxPreProcessing

    img = cv2.imread(image_path)

    # Pre-Processing
    letterboxing = LetterboxPreProcessing(color=color, auto=False)
    img_transformed, ratio, _ = letterboxing(img, new_shape=letterbox_size)

    if enable_bgr_to_rgb:
        img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    img_transformed = img_transformed[None]  # HWC -> NHWC

    return img_transformed, ratio
