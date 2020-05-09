import cv2
import numpy

from lib.imgparse import parse_image
from lib.imgproc import crop_rect
from lib.profiles import PARSES, Parsetype, Profile
from lib.utils import dump


def parse_buffs(src: numpy.ndarray, profile: Profile):
    frame: numpy.ndarray = src
    frame = crop_rect(src, profile.game_crop)
    frame = crop_rect(frame, PARSES[Parsetype.Buffs].crop_coords)
    scale_factor = 64 / frame.shape[0]
    frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), 64))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.normalize(frame, frame, 256, 0, cv2.NORM_MINMAX)
    frame = cv2.adaptiveThreshold(frame, frame.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 999, -50)

    text = parse_image(frame)
    return text
