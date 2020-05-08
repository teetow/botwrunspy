from typing import List

import cv2
import numpy

from lib.imgparse import ParseMode, parse_image
from lib.imgproc import conform_color, crop_rect, threshold
from lib.profiles import Profile
from lib.split import Split
from lib.utils import dump

IDEAL_HEIGHT = 640


def conform_default(frame: numpy.ndarray) -> numpy.ndarray:
    pass


def parse_splits(src: numpy.ndarray, profile: Profile) -> List[Split]:
    splits = []
    frame = crop_rect(src, profile.split_crop, profile.postcrop_func or None)
    if frame.shape[0] < IDEAL_HEIGHT:
        scale_factor = IDEAL_HEIGHT / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), IDEAL_HEIGHT), cv2.INTER_LANCZOS4)

    frame = conform_color(frame, profile.conform_func)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.bitwise_not(frame)
    frame = frame*1.3 - 0.7
    text = parse_image(frame, ParseMode.BlockUniform)

    for line in text.split("\n"):
        split = Split.fromRaw(line)
        if split:
            splits.append(split)
    return splits
