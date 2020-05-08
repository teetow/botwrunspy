import cv2


def dump(f):
    cv2.imwrite("testdata/out.png", f)


def show(f):
    cv2.imshow("f", f)
    cv2.waitKey()
