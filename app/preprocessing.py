
import numpy as np
from scipy.signal import savgol_filter

def normalize_landmarks(seq):
    out = []
    for lm in seq:
        if np.isnan(lm).all():
            out.append(lm)
            continue
        center = np.nanmean(lm[:, :2], axis=0)
        scale = np.nanstd(lm[:, :2]) or 1.0
        n = lm.copy()
        n[:, :2] = (n[:, :2] - center) / scale
        out.append(n)
    return out

def compute_angles_sequence(seq):
    res = {}
    for i, lm in enumerate(seq):
        for j in range(len(lm)):
            res.setdefault(j, []).append(lm[j, 0])
    return {k: np.array(v) for k, v in res.items()}

def smooth_series(series, window=11, poly=3):
    out = {}
    for k, v in series.items():
        if len(v) >= window:
            out[k] = savgol_filter(v, window, poly)
        else:
            out[k] = v
    return out
