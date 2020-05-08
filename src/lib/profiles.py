from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, List, Tuple

import cv2
import numpy

from lib.imgproc import Channelmap
from lib.rect import Rect


@dataclass
class Profile():
    url: str
    conform_func: Callable[[numpy.ndarray], numpy.ndarray]
    split_crop: Rect = None
    game_crop: Rect = None
    postcrop_func: Callable[[numpy.ndarray], numpy.ndarray] = None


class Names(IntEnum):
    Default = auto()
    Specs = auto()
    DJ = auto()
    JoeDunff = auto()


def dj_conform_func(m: Channelmap):
    out = (
        m.r * 0.24
        + m.g * 1.4
        - m.b * 0.5
        - m.c * 0.15
    )
    return numpy.clip(out, 0, 256)


PROFILES = {
    Names.Specs: Profile(
        "https://www.twitch.tv/specsnstats",
        lambda m: (m.r * 1.5) - (m.c * 1.5) + (m.b * 1.0) - 25,
        Rect(0.054, 0.439, 0.002, 0.211),  # splits
        Rect(0.104, 0.896, 0.214, 0.998)  # game
    ),

    Names.DJ: Profile(
        "https://www.twitch.tv/djcarmichael",
        dj_conform_func,
        Rect(0.178, 0.409, 0.823, 0.991),  # splits
        Rect(0.014, 0.821, 0.011, 0.816)  # game
    ),

    Names.JoeDunff: Profile(
        "https://www.twitch.tv/joedunff",
        lambda m: ((m.g * 1.0 + m.b * 1.0) - 160),
        Rect(0.077, 0.513, 0.003, 0.196),  # splits
        Rect(0.107, 0.894, 0.202, 0.99)  # game
    ),
}


class Parsetype(IntEnum):
    NoParse = 0
    Gameframe = auto()
    Bossname = auto()
    Bosstype = auto()


@dataclass
class ParseOp():
    crop_coords: Rect


PARSES = {
    Parsetype.Bossname: ParseOp(
        Rect(0.034, 0.071, 0.32, 0.681),
    ),
    Parsetype.Bosstype: ParseOp(
        Rect(0.066, 0.129, 0.32, 0.681)
    )

}
