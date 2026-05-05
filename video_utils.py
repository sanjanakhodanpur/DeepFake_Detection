# video_utils.py
import os
import cv2
import numpy as np
from typing import List

# Supported file extensions (lowercase)
SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_IMAGE_EXT


def is_video_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_VIDEO_EXT


def load_image_frames(path: str) -> List[np.ndarray]:
    """
    Load an image and return a list containing one RGB frame.
    The caller will duplicate/pad frames if needed for the LSTM.
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"Unable to read image file: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return [img_rgb]


def load_video_frames(path: str, max_frames: int | None = None, every_n: int = 3) -> List[np.ndarray]:
    """
    Load frames from a video using OpenCV.

    Args:
        path: path to video file
        max_frames: maximum number of frames to keep.
                    If None, read until the end of the video.
        every_n: keep every Nth frame to reduce computation.
                 For 10–20s videos, 2 or 3 is usually fine.

    Returns:
        List of RGB frames (numpy arrays)
    """
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(
                "Unable to open video file. The codec may be unsupported. "
                "Please use a standard MP4 (H.264) or AVI file."
            )

    frames: List[np.ndarray] = []
    frame_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # only keep every_n-th frame
        if every_n > 1 and frame_index % every_n != 0:
            frame_index += 1
            continue

        if frame_bgr is None or frame_bgr.size == 0:
            frame_index += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_index += 1

        # stop if we hit max_frames (when not None)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise RuntimeError(
            "No readable frames extracted from the video. "
            "The file may be corrupted or encoded with an unsupported codec."
        )

    return frames
