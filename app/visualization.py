
import numpy as np
import cv2

def draw_skeleton(frame, lm):
    return frame

def make_side_by_side(user, ref, bad, err):
    h = max(user.shape[0], ref.shape[0])
    user = cv2.resize(user, (user.shape[1], h))
    ref = cv2.resize(ref, (ref.shape[1], h))
    return np.hstack([user, ref])
