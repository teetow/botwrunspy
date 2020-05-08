import os
import time

import cv2
import numpy
from streamlink import Streamlink

streamer = Streamlink()


def grabframes(url: str, limit=1e3) -> numpy.ndarray:
    plugin = streamer.resolve_url(url)
    stream = plugin.streams().get("best")
    if stream is None:
        print("Failed to get stream. Try again.")
        return
    fd = stream.open()
    data = bytearray()
    while len(data) < 1e6:
        data += fd.read(1024)
    fname = "stream.bin"
    with open(fname, "wb") as outfile:
        outfile.write(data)

    capture = cv2.VideoCapture(fname)
    imgs = []
    while capture.isOpened():
        img = capture.read()[1]
        if img is None:
            break
        imgs.append(img)
        if len(imgs) >= limit:
            break
    capture.release()
    os.remove("stream.bin")
    return imgs


def grabframe(url: str) -> numpy.ndarray:
    return grabframes(url, 1)[0]


def test():
    f = grabframe("https://www.twitch.tv/underthebed")
    if f is not None:
        cv2.imwrite("data/underthebed.png", f)
