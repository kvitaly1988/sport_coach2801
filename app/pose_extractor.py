
from __future__ import annotations
import numpy as np
import mediapipe as mp

def _require_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        raise ImportError(
            "OpenCV недоступен. Используйте opencv-python-headless и Python 3.11. "
            f"Ошибка: {e}"
        )

mp_pose = mp.solutions.pose

def get_video_frames(path: str, max_width: int = 960):
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        frames.append(frame)
    cap.release()
    return fps, frames

def extract_pose_from_video(path: str):
    cv2 = _require_cv2()
    fps, frames = get_video_frames(path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
    landmarks = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            arr = np.array([[p.x, p.y, p.visibility] for p in res.pose_landmarks.landmark], dtype=np.float32)
        else:
            arr = np.full((33, 3), np.nan, np.float32)
        landmarks.append(arr)
    pose.close()
    return {"fps": fps, "frames": frames, "landmarks": landmarks}
