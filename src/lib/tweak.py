import cv2

from lib.imgproc import Channelmap
from lib.utils import dump
import numpy as np

WINNAME = "Tweakme"

MAX = 128

def tweak(m: Channelmap):
    sliders = {
        "r": 1.0,
        "g": 1.0,
        "b": 1.0,
        "c": 0.0,
        "m": 0.0,
        "y": 0.0,
        "k": 0.0,
        "gain": 1.0,
        "offset": 0.0,
    }
    slider_ranges = {
        "r": (MAX, MAX),
        "g": (MAX, MAX),
        "b": (MAX, MAX),
        "c": (MAX // 2, MAX),
        "m": (MAX // 2, MAX),
        "y": (MAX // 2, MAX),
        "k": (MAX // 2, MAX),
        "gain": (MAX // 2, MAX),
        "offset": (MAX // 2, MAX),
    }

    def on_trackbar(bar_name, value):
        sliders[bar_name] = (value - MAX // 2) / (MAX // 2)
        frame = (
            (
                (
                    m.r * sliders["r"] +
                    m.g * sliders["g"] +
                    m.b * sliders["b"] +
                    m.c * sliders["c"] +
                    m.m * sliders["m"] +
                    m.y * sliders["y"] +
                    m.k * sliders["k"]
                )
                * (sliders["gain"] + 1.0)
            )
            + sliders["offset"] * MAX // 2
        )
        out = np.clip(frame, 0, 255)
        out = out.astype("uint8")
        print(sliders)
        cv2.imshow(WINNAME, out)

    cv2.namedWindow(WINNAME)

    def make_cb(slider_name):
        def callback(val):
            on_trackbar(slider_name, val)
        return callback

    for slider in sliders:
        cb = make_cb(slider)
        default, max_val = slider_ranges[slider]
        cv2.createTrackbar(f"bar_{slider}", WINNAME, default, max_val, cb)

    on_trackbar("init", 0)
    while True:
        ret = cv2.waitKey()
        if ret in [13, 27]:
            break
    pass
