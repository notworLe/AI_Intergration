import cv2
import numpy as np


def apply_transform(frame, name, param):
    """
    frame : BGR numpy array từ cv2
    name  : tên transform (string)
    param : giá trị 1-100 từ slider
    """
    h, w = frame.shape[:2]

    if name == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif name == "blur":
        k = param // 10 * 2 + 1  # kernel phải lẻ
        return cv2.GaussianBlur(frame, (k, k), 0)

    elif name == "canny":
        lo = param * 2
        hi = lo * 3
        edges = cv2.Canny(frame, lo, hi)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif name == "rotate":
        angle = param * 3.6  # 0-360 độ
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))

    elif name == "flip":
        return cv2.flip(frame, 1)  # horizontal

    elif name == "threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_val = int(param * 2.55)  # 0-255
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    elif name == "perspective":
        offset = param * 2
        src = np.float32([
            [0, 0], [w, 0],
            [0, h], [w, h]
        ])
        dst = np.float32([
            [offset, offset], [w - offset, 0],
            [0, h - offset], [w, h]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (w, h))

    elif name == "resize":
        scale = param // 50
        return cv2.resize(frame.copy(), None, fx=scale, fy=scale)

    return frame  # "none" — giữ nguyên