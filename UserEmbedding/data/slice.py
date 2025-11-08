import os
import re
import cv2
import json
import glob
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch

from decord import VideoReader, cpu
import jax
import jax.numpy as jnp
from videoprism import models as vp

ORIGINAL_FPS = 60  # original fps of videos in dataset
VIDEO_WIDTH = 288
VIDEO_HEIGHT = 288
VIDEOPRISM_MODEL_NAME = 'videoprism_public_v1_base'

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def slice_video(
    video_path: str,
    length_frames: int,
    step: int,
    output_dir: str,
    videoprism_model,
    use_bfloat16: bool = True,
) -> int:
    ensure_dir(output_dir)
    
    fprop_dtype = jnp.bfloat16 if use_bfloat16 else None
    flax_model = vp.get_model(videoprism_model, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(videoprism_model)
    
    vr = VideoReader(video_path, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, ctx=cpu(0))
    T = len(vr)

    start = 0
    idx = 0

    basename = os.path.splitext(os.path.basename(video_path))[0]

    while start <= T - step * length_frames:
        inds = list(range(start, start + step * length_frames, step))
        batch = vr.get_batch(inds).asnumpy()
        batch = torch.from_numpy(batch).unsqueeze(0).numpy()
        batch = jnp.asarray(batch, dtype=fprop_dtype or jnp.float32)
        
        embeddings, _ = flax_model.apply(loaded_state, batch, train=False)
        embeddings = embeddings.squeeze(0)  # [T, D]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        np.save(os.path.join(output_dir, f"{basename}_slice{idx}.npy"), embeddings)
        
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
        
    return idx


def slice_dataset(
    video_dir,
    pose_estimation_dir,
    length_frames: int = 243,
    fps: int = None,
):
    """
    Slices each (video, pose) pair into aligned 243-frame windows.
    """
    vid_out = video_dir + "_embedding_sliced"
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
            videoprism_model=VIDEOPRISM_MODEL_NAME,
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
