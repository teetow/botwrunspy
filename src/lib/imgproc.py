import time
from dataclasses import dataclass

import cv2
import numpy

from lib.rect import Rect
from lib.utils import dump

MAX = 255


@dataclass
class Channelmap():
    r: numpy.ndarray
    g: numpy.ndarray
    b: numpy.ndarray
    c: numpy.ndarray
    m: numpy.ndarray
    y: numpy.ndarray
    k: numpy.ndarray

    @staticmethod
    def from_img(img: numpy.ndarray):
        b, g, r = cv2.split(img)
        c, m, y, k = get_cmyk(r, g, b)
        return Channelmap(r, g, b, c, m, y, k)


def get_cmyk(r, g, b):

    def complement(ch):
        inv_ch = MAX - ch
        inv_k = MAX - k
        return ((inv_ch - k) / inv_k) * MAX

    k = 0.01 + cv2.min(cv2.min(MAX - r, MAX - g), MAX - b) * 0.99
    c = complement(r)
    m = complement(g)
    y = complement(b)
    return c, m, y, k


def conform_color(img, conform_func, postcrop_func=None):
    m = Channelmap.from_img(img)
    v = conform_func(m)
    v = cv2.normalize(v, None, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
    out = cv2.merge((v, v, v))
    return out


def threshold(img, c=-10):
    mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thres_mode = cv2.THRESH_BINARY_INV
    blocksize = 999
    img = cv2.adaptiveThreshold(img, img.max(), mode, thres_mode, blocksize, c)
    return img


def crop(img, ys, ye, xs, xe, postcrop_func=None):
    img = img[ys:ye, xs:xe]

    if postcrop_func:
        img = postcrop_func(img)

    return img


def crop_rect(img, rect: Rect, postcrop_func=None):
    ys, ye, xs, xe = rect.scaled(img.shape[1], img.shape[0]).as_tuple()
    return crop(img, ys, ye, xs, xe, postcrop_func)
