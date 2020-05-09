from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, List, Tuple

import cv2
import numpy

from lib.imgproc import Channelmap
from lib.rect import Rect
from lib.tweak import tweak
from lib.utils import dump


@dataclass
class Profile():
    """
    Metadata for a stream.
    URL -- Twitch URL
    conform_func -- a function that outputs white-on-black text
    split_crop: fractional coordinates of current splits
    game_crop: fractional coordinates of game frame
    postcrop_func: postprocessing step called after cropping splits
    """
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
    JohnnyBoomr = auto()


def conform_default(m: Channelmap):
    frame = m.k.astype("uint8")
    return frame


def conform_specs_splits(m: Channelmap):
    vals = {'r': 0.359375, 'g': 0.0625, 'b': 0.25, 'c': -0.265625, 'm': 0.0, 'y': -1.0, 'k': -0.09375, 'gain': 1.5, 'offset': -0.9, 'init': -1.0}
    frame = (
        (
            m.r * vals['r'] +
            m.g * vals['g'] +
            m.b * vals['b'] +
            m.c * vals['c'] +
            m.m * vals['m'] +
            m.y * vals['y'] +
            m.k * vals['k']
        ) * (vals['gain'] + 1.0) 
        + vals['offset'] * 128
    )
    frame = numpy.clip(frame, 0, 255)
    frame = frame.astype("uint8")
    frame = cv2.adaptiveThreshold(frame, frame.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 999, -20)
    return frame


def conform_joe_splits(m: Channelmap):
    frame = ((m.g * 1.0 + m.b * 1.0) * 0.76 - 130)
    frame = 256 - numpy.clip(frame, 0, 256)
    return frame


def conform_dj_splits(m: Channelmap):
    frame = (
        m.r * 0.24
        + m.g * 1.4
        - m.b * 0.5
        - m.c * 0.15
    )
    return numpy.clip(frame, 0, 256)


PROFILES = {
    Names.Specs: Profile(
        "https://www.twitch.tv/specsnstats",
        conform_specs_splits,
        Rect(0.054, 0.439, 0.002, 0.211),  # splits
        Rect(0.104, 0.896, 0.214, 0.998)  # game
    ),

    Names.DJ: Profile(
        "https://www.twitch.tv/djcarmichael",
        conform_dj_splits,
        Rect(0.178, 0.409, 0.823, 0.991),  # splits
        Rect(0.014, 0.821, 0.011, 0.816)  # game
    ),

    Names.JoeDunff: Profile(
        "https://www.twitch.tv/joedunff",
        conform_joe_splits,
        Rect(0.077, 0.513, 0.003, 0.196),  # splits
        Rect(0.107, 0.894, 0.202, 0.99)  # game
    ),
    Names.JohnnyBoomr: Profile(
        "https://www.twitch.tv/johnnyboomr",
        conform_default,
        Rect(0.313, 0.519, 0.8, 0.999),
        Rect.full(),
    )
}


class Parsetype(IntEnum):
    NoParse = 0
    Gameframe = auto()
    Bossname = auto()
    Bosstype = auto()
    Buffs = auto()


@dataclass
class ParseOp():
    crop_coords: Rect


PARSES = {
    Parsetype.Bossname: ParseOp(
        Rect(0.034, 0.071, 0.32, 0.681),
    ),
    Parsetype.Bosstype: ParseOp(
        Rect(0.066, 0.129, 0.32, 0.681)
    ),
    Parsetype.Buffs: ParseOp(
        Rect(0.119, 0.158, 0.159, 0.316)
    )
}
