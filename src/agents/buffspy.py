from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, auto

import cv2
import numpy
from parse import parse

from lib.imgparse import parse_image
from lib.imgproc import crop_rect
from lib.profiles import PARSES, Parsetype, Profile
from lib.utils import dump


class BuffType(IntEnum):
    Unknown = auto()
    SpeedUp = auto()


@dataclass
class Buff():
    buff_type: BuffType
    numSeconds: int

    @property
    def timestamp(self):
        return f"{self.numSeconds // 60}:{self.numSeconds % 60}"

    @classmethod
    def from_strs(cls, typestr: str, timestr: str) -> Buff:
        bufftype = BuffType.SpeedUp if typestr.lower() == "speed up" else BuffType.Unknown
        mins, secs = parse("{:d}:{:d}", timestr)
        numsecs = mins*60+secs
        return Buff(bufftype, numsecs)

    def __str__(self):
        return f"{self.buff_type.name} {self.timestamp}"


re_buff = re.compile(r"(?P<type>[\w ]+).+?(?P<ts>\d{2}:\d{2})")


def parse_buffs(src: numpy.ndarray, profile: Profile):
    frame: numpy.ndarray = src
    frame = crop_rect(frame, PARSES[Parsetype.Buffs].crop_coords)
    scale_factor = 64 / frame.shape[0]
    frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), 64))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.normalize(frame, frame, 256, 0, cv2.NORM_MINMAX)
    frame = cv2.adaptiveThreshold(frame, frame.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 999, -50)

    text = parse_image(frame)
    match = re_buff.match(text)

    if match is not None:
        return Buff.from_strs(match.groupdict()["type"].strip(), match.groupdict()["ts"])
