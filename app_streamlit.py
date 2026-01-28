
import json, os, tempfile
import streamlit as st
import numpy as np

from app.pose_extractor import extract_pose_from_video
from app.visualization import draw_skeleton, make_side_by_side

st.set_page_config(layout="wide")
st.title("AI‑коуч: покадровое сравнение (Cloud-ready)")

with open("app/elements_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

el = st.selectbox("Элемент", list(cfg.keys()), format_func=lambda k: cfg[k]["title"])

user_file = st.file_uploader("Видео пользователя", type=["mp4","avi","mov"])

if user_file and st.button("Анализировать"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(user_file.read())
    tmp.close()

    user = extract_pose_from_video(tmp.name)
    ref = extract_pose_from_video(cfg[el]["reference_video"])

    st.success("Проект успешно запущен в Streamlit Cloud.")
    st.info("Это базовая Cloud-версия. Расширенный анализ можно включить поэтапно.")
