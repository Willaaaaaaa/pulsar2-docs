from typing import Tuple

import cv2

import numpy as np


class LetterboxCenterCropPreProcessing:
    def __init__(
        self,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        add_tophat: bool = True,
    ) -> None:

        self.crop_size = crop_size
        self.new_size = resize_size
        self.mean = list(mean)
        self.std = list(std)
        self.add_tophat = add_tophat

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]  # current shape
        if height < width:
            h0 = self.new_size
            w0 = int(width * (self.new_size / height))
        else:
            h0 = int(height * (self.new_size / width))
            w0 = self.new_size
        image = cv2.resize(image, (w0, h0), interpolation=cv2.INTER_LINEAR)

        center_h = h0 // 2
        center_w = w0 // 2
        image = image[
            center_h - self.crop_size // 2 : center_h + self.crop_size // 2,
            center_w - self.crop_size // 2 : center_w + self.crop_size // 2,
        ]

        return image
