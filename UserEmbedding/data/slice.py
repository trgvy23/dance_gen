import os
import re
import cv2
import json
import glob
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

from decord import VideoReader, cpu

ORIGINAL_FPS = 60  # original fps of videos in dataset


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def slice_video(
    video_path: str,
    length_frames: int,
    step: int,
    output_dir: str,
    width: int,
    height: int,
) -> int:
    if width is not None and height is not None:
        vr = VideoReader(video_path, width=width, height=height)  # (T, H, W, 3)
    else:
        vr = VideoReader(video_path)  # (T, H, W, 3)
    T = len(vr)

    start = 0
    idx = 0

    basename = os.path.splitext(os.path.basename(video_path))[0]

    while start <= T - step * length_frames:
        inds = list(range(start, start + step * length_frames, step))
        batch = vr.get_batch(inds).asnumpy()
        np.save(os.path.join(output_dir, f"{basename}_slice{idx}.npy"), batch)
        start += length_frames
        idx += 1

    return idx


def slice_motion_estimation(
    pose_path: str, length_frames: int, step: int, output_dir: str
) -> int:
    with open(pose_path, "r") as read_file:
        pose_ests = json.load(read_file)

    T = len(pose_ests)

    start = 0
    idx = 0

    basename = os.path.splitext(os.path.basename(pose_path))[0]

    while start <= T - step * length_frames:
        slice_ests = pose_ests[start : start + step * length_frames : step]
        with open(
            os.path.join(output_dir, f"{basename}_slice{idx}.json"), "w"
        ) as write_file:
            json.dump(slice_ests, write_file)
        start += length_frames
        idx += 1


def slice_dataset(
    video_dir,
    pose_estimation_dir,
    length_frames: int = 243,
    width: int = None,
    height: int = None,
    fps: int = None,
):
    """
    Slices each (video, pose) pair into aligned 243-frame windows.
    """
    vid_out = video_dir + "_sliced"
    pose_est_out = pose_estimation_dir + "_sliced"

    ensure_dir(vid_out)
    ensure_dir(pose_est_out)

    video_files = sorted(glob.glob(f"{video_dir}/*.mp4"))
    pose_est_files = sorted(glob.glob(f"{pose_estimation_dir}/*.json"))
    assert len(video_files) == len(
        pose_est_files
    ), "Mismatch between video and pose estimation file counts"

    data_stride = ORIGINAL_FPS // fps if fps is not None else 1

    for video_path, pose_path in tqdm(zip(video_files, pose_est_files), desc="Slicing"):
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        pose_basename = os.path.splitext(os.path.basename(pose_path))[0]

        assert video_basename == pose_basename, str((video_basename, pose_basename))

        video_slices = slice_video(
            video_path=video_path,
            length_frames=length_frames,
            step=data_stride,
            output_dir=vid_out,
            width=width,
            height=height,
        )
        pose_slices = slice_motion_estimation(
            pose_path=pose_path,
            length_frames=length_frames,
            step=data_stride,
            output_dir=pose_est_out,
        )
        # make sure the slices line up
        assert video_slices == pose_slices, str(
            (video_path, pose_path, video_slices, pose_slices)
        )
