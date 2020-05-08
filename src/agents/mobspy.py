from typing import Tuple

import cv2
import numpy

from lib.framegrab import grabframes
from lib.imgparse import ParseMode, parse_image
from lib.imgproc import conform_color, crop
from lib.profiles import PARSES, ParseOp, Parsetype


def preprocess(f: numpy.ndarray, p: ParseOp) -> numpy.ndarray:
    crop_rect = p.crop_coords.scaled(f.shape[1], f.shape[0])
    f = crop(f, *(crop_rect.as_tuple()))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    print(f.max(), f.min())
    f = cv2.normalize(f, f, 256, 0, cv2.NORM_MINMAX)
    f = cv2.adaptiveThreshold(f, f.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 999, -50)
    return f


def get_lines(f):
    s = parse_image(f, ParseMode.BlockUniform)

    if s is None or s.strip() == "":
        return

    return [x.strip() for x in s.splitlines()]


def get_name(f):
    p: ParseOp = PARSES[Parsetype.Bossname]
    f = preprocess(f, p)
    lines = get_lines(f)
    return lines


def get_type(f):
    p: ParseOp = PARSES[Parsetype.Bosstype]
    f = preprocess(f, p)
    lines = get_lines(f)
    return lines


def parse_mob(game_frame: numpy.ndarray) -> Tuple[str]:
    """
    Extract mob name and type, if present in the screenshot.

    Arguments:
    game_frame -- image data (just the game frame, no streamer UI)

    """
    bosstype = get_type(game_frame)
    bossname = get_name(game_frame)
    return (bosstype, bossname)
