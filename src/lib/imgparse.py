from enum import IntEnum

import numpy
import pytesseract
from PIL import Image


def parse_file(filename) -> str:
    return pytesseract.image_to_string(Image.open(filename))

#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR.
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
# 			bypassing hacks that are Tesseract-specific.


class ParseMode(IntEnum):
    NoMode = 0
    Default = 3
    SingleColumn = 4
    BlockValign = 5
    BlockUniform = 6
    SingleLine = 7
    Sparse = 11
    SparseOsd = 12
    Raw = 13


def parse_image(img: numpy.ndarray, mode: ParseMode = ParseMode.NoMode) -> str:
    config = ""
    if mode:
        config += f"--psm {mode}"
    return pytesseract.image_to_string(Image.fromarray(img), config=config)
