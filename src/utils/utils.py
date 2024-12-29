from functools import wraps
from pathlib import Path
from time import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_img(filepath: Path) -> np.ndarray:
    return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)  # type: ignore


def show_img(img: np.ndarray, title: str | None = None):
    plt.imshow(img, interpolation="nearest")
    plt.gray()
    if title:
        plt.title(title)
    plt.show()


def show_img_cv(img: np.ndarray):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap
