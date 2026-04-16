import os
import re
import cv2
import json
import glob
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import torch
import mediapy
import pickle
import shutil
from decord import VideoReader, cpu
import jax
import jax.numpy as jnp
from videoprism import models as vp
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)
import pickle
import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm

from extracting_features.video_mask_features import get_video_masks_features
from extracting_features.video_features import extract_video_features
from extracting_features.audio_baseline_features import extract_folder
from extracting_features import alphapose_features
from extracting_features import motionbert_features
from io_files import ensure_dir

ORIGINAL_FPS = 60  # original fps of videos in dataset
VIDEO_WIDTH = 288
VIDEO_HEIGHT = 288
VIDEOPRISM_MODEL_NAME = "videoprism_public_v1_base"


def build_segmentation_model(device: str = "cuda"):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    class_to_idx = {cls: idx for (idx, cls) in enumerate(
        weights.meta["categories"])}
    person_idx = class_to_idx["person"]  # human class

    return model, preprocess, person_idx


def slice_audio(
    audio_path: str,
    output_dir: str,
    n_frames: int,
    fps: int,
    overlap: float = 1.0,
) -> list:
    """
    Slice audio into chunks corresponding to a number of video frames.

    Args:
        audio_path: Path to input audio file.
        output_dir: Directory to save audio slices.
        n_frames: Number of video frames per slice.
        fps: Frame rate of the video (used to calculate duration).
        overlap: Fraction of segment to move forward
                 1.0 = no overlap
                 0.5 = overlap half of previous segment

    Returns:
        List of audio slices.
    """
    ensure_dir(output_dir)
    audio, sr = lr.load(audio_path, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Calculate samples per window and step
    # Duration (s) = n_frames / fps
    samples_per_window = int(round(n_frames * sr / fps))
    step_samples = int(round(n_frames * sr / fps * overlap))
    total_samples = len(audio)

    start_idx = 0
    idx = 0
    list_sliced_audio = []

    # print(f"Slicing audio {audio_path}...")
    # print(f"\tSample rate: {sr}, FPS: {fps}")
    # print(f"\tWindow: {n_frames} frames -> {samples_per_window} samples")
    # print(f"\tStep: {overlap} -> {step_samples} samples")

    # Iterate while the start of the slice is within the limit
    while start_idx < total_samples:
        # print(
        #     f"\tSlice {idx} from {start_idx} to {min(start_idx + samples_per_window, total_samples)}"
        # )

        # Determine strict end if we want to mimic video 'break' behavior?
        # In video: break if start + n_frames >= total_frames
        # That means the last slice must be 'started' before the end, but if it goes over, it's padded.
        # But if the next start is >= total, we stop.
        # Here: start_idx < limit_samples handles that.

        end_idx = min(start_idx + samples_per_window, total_samples)
        audio_slice = audio[start_idx:end_idx]

        # Pad with zeros if the slice is shorter than expected (last slice)
        if len(audio_slice) < samples_per_window:
            # print(f"\t\tMissing audio at segment {start_idx} - {end_idx}")
            pad_len = samples_per_window - len(audio_slice)
            audio_slice = np.pad(audio_slice, (0, pad_len), mode="constant")

        out_path = os.path.join(output_dir, f"{file_name}_slice{idx}.wav")
        sf.write(out_path, audio_slice, sr)
        assert os.path.exists(
            out_path), f"Failed to create audio slice: {out_path}"
        # print(f"\tCreated audio slice: {out_path}")

        start_idx += step_samples
        idx += 1
        list_sliced_audio.append(out_path)

    return list_sliced_audio


def extract_baseline_feat_sliced_audio(
    sliced_audio_dir: str,
    baseline_feat_audio_dir: str,
) -> list:
    ensure_dir(baseline_feat_audio_dir)
    extract_folder(sliced_audio_dir, baseline_feat_audio_dir)
    return [
        os.path.join(baseline_feat_audio_dir, f)
        for f in os.listdir(baseline_feat_audio_dir)
        if f.endswith(".npy")
    ]


def extract_jukebox_feat_sliced_audio(
    sliced_audio_dir: str,
    jukebox_feat_audio_dir: str,
) -> list:
    ensure_dir(jukebox_feat_audio_dir)
    extract_folder(sliced_audio_dir, jukebox_feat_audio_dir)
    return [
        os.path.join(jukebox_feat_audio_dir, f)
        for f in os.listdir(jukebox_feat_audio_dir)
        if f.endswith(".npy")
    ]


def slice_video(
    video_path: str,
    output_dir: str,
    n_frames: int,
    target_fps: int,
    overlap: float = 1.0,
) -> list:
    """
    Slice a video into fixed-length segments.

    Args:
        video_path: path to input video
        output_dir: directory to save video segments
        n_frames: number of frames per segment
        target_fps: fps to sample the video
        overlap: fraction of segment to move forward
                 1.0 = no overlap
                 0.5 = overlap half of previous segment
        pad_value: pixel value for padding (default: 0 = black)

    Returns:
        list of video slices
    """
    assert 0 < overlap <= 1.0, "overlap must be in (0, 1]"

    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame sampling stride to reach target fps
    fps_stride = max(int(round(src_fps / target_fps)), 1)

    # Segment stride (controls overlap)
    segment_stride = max(int(round(n_frames * overlap)), 1)

    frames = []
    frame_idx = 0
    PAD_VALUE = 0

    # --- Read & sample frames ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps_stride == 0:
            frames.append(frame)

        frame_idx += 1

    cap.release()

    total_frames = len(frames)
    slice_videos_list = []
    segment_count = 0

    # --- Slice into segments ---
    # print(f"Slicing video {video_path}")
    for start in range(0, total_frames, segment_stride):
        # print(f"\tSlice from frame {start} to {start + n_frames}")
        segment = frames[start: start + n_frames]
        if len(segment) == 0:
            break

        # Pad last segment if needed
        if len(segment) < n_frames:
            # print(f"\t\tMissing frame at segment {start} - {start + n_frames}")
            pad_len = n_frames - len(segment)
            pad_frame = np.full(
                (height, width, 3),
                PAD_VALUE,
                dtype=np.uint8,
            )
            segment.extend([pad_frame] * pad_len)

        out_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_slice{segment_count}.mp4",
        )
        slice_videos_list.append(out_path)

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            (width, height),
        )

        for f in segment:
            writer.write(f)

        writer.release()
        assert os.path.exists(
            out_path), f"Failed to create video slice: {out_path}"
        segment_count += 1

        # Stop if this was the padded last segment
        if start + n_frames >= total_frames:
            break

    return slice_videos_list


def slice_motion_estimation(
    pose_path: str, length_frames: int, step: int, output_dir: str
) -> int:
    with open(pose_path, "r") as read_file:
        pose_ests = json.load(read_file)

    T = len(pose_ests)

    # print(f"Slicing motion estimation {pose_path} with {T} frames...")
    start = 0
    idx = 0
    basename = os.path.splitext(os.path.basename(pose_path))[0]

    slice_length = step * length_frames
    padding_keypoints = [0] * 78
    padding_keypoints = {
        "keypoints": padding_keypoints,
    }
    while start < T:
        # print(f"Slicing motion estimation {idx} from frame {start}...")
        slice_ests = pose_ests[start: min(start + slice_length, T): step]
        if len(slice_ests) < slice_length:
            cur = len(slice_ests)
            pad = slice_length - cur
            slice_ests.extend([padding_keypoints] * pad)
        with open(
            os.path.join(output_dir, f"{basename}_slice{idx}.json"), "w"
        ) as write_file:
            json.dump(slice_ests, write_file)
        start += slice_length
        idx += 1

    return idx


def slice_motion(
    motion_path: str,
    output_dir: str,
    n_frames: int,
    target_fps: int,
    overlap: float = 1.0,
    original_fps: int = ORIGINAL_FPS,
) -> list[str]:
    """
    Slice motion into fixed-length segments

    Args:
        motion_path: path to motion .pkl
        output_dir: directory to save motion slices
        n_frames: number of frames per segment (after downsampling)
        target_fps: fps to sample motion to
        overlap: fraction of segment to move forward
                 1.0 = no overlap
                 0.5 = overlap half of previous segment
        original_fps: original fps of motion data

    Returns:
        list of motion slice paths
    """
    assert 0 < overlap <= 1.0, "overlap must be in (0, 1]"

    ensure_dir(output_dir)

    with open(motion_path, "rb") as f:
        motion_data = pickle.load(f)

    keys = motion_data.keys()
    T = len(motion_data["smpl_poses"])

    # Downsampling stride (same idea as fps_stride in video)
    data_stride = max(int(round(original_fps / target_fps)), 1)

    # Segment stride (controls overlap)
    segment_stride = max(int(round(n_frames * overlap)), 1)

    # Total raw frames consumed per segment
    slice_length = n_frames * data_stride

    basename = os.path.splitext(os.path.basename(motion_path))[0]

    # print(f"Slicing motion {motion_path}")
    # print(f"  Total frames: {T}")
    # print(f"  data_stride: {data_stride}")
    # print(f"  segment_stride: {segment_stride}")

    slice_paths = []
    segment_count = 0

    for start in range(0, T, segment_stride * data_stride):
        # print(f"\tSlice from frame {start} to {start + slice_length}")

        sliced_motion = {}

        for key in keys:
            if key == "smpl_loss":
                sliced_motion[key] = 0.0
            elif key == "smpl_scaling":
                sliced_motion[key] = motion_data[key]
            else:
                chunk = motion_data[key][
                    start: min(start + slice_length, T): data_stride
                ]

                # Padding if needed
                if len(chunk) < n_frames:
                    pad_len = n_frames - len(chunk)

                    if isinstance(chunk, np.ndarray):
                        padding = np.zeros(
                            (pad_len,) + chunk.shape[1:],
                            dtype=chunk.dtype,
                        )
                        chunk = np.vstack([chunk, padding])
                    else:
                        chunk = list(chunk)
                        chunk.extend([0] * pad_len)

                sliced_motion[key] = chunk

        out_path = os.path.join(
            output_dir,
            f"{basename}_slice{segment_count}.pkl",
        )
        with open(out_path, "wb") as f:
            pickle.dump(sliced_motion, f)

        assert os.path.exists(
            out_path), f"Failed to create motion slice: {out_path}"
        slice_paths.append(out_path)
        segment_count += 1

        # Stop after padded last segment (same logic as video)
        if start + slice_length >= T:
            break

    return slice_paths


def slice_dataset(
    data_dir: str,
    data_list: list,
    length_frames: int = 243,
    fps: int = None,
    overlap: float = 1.0,
    skip_if_exists: bool = False,
):
    """
    Slices each (video, pose) pair into aligned 243-frame windows.
    """
    vid_data_dir = os.path.join(data_dir, "video")
    assert os.path.exists(vid_data_dir), f"Path ${vid_data_dir} not found"
    music_data_dir = os.path.join(data_dir, "wavs")
    assert os.path.exists(music_data_dir), f"Path ${music_data_dir} not found"
    motion_data_dir = os.path.join(data_dir, "motions")
    assert os.path.exists(
        motion_data_dir), f"Path ${motion_data_dir} not found"
    pbar = tqdm(data_list, desc="Slicing dataset")

    # for data in data_list:
    total_sliced_data = []
    for data in pbar:
        video_path = os.path.join(vid_data_dir, data + ".mp4")
        music_path = os.path.join(music_data_dir, data + ".wav")
        motion_path = os.path.join(motion_data_dir, data + ".pkl")
        pbar.set_postfix_str(f"Slicing {data}...")

        audios_sliced_list, videos_sliced_list, motions_sliced_list = slice_single_tuple(
            data_dir,
            video_path,
            music_path,
            motion_path,
            length_frames,
            fps,
            overlap,
            skip_if_exists
        )
        total_sliced_data.extend([os.path.splitext(os.path.basename(video_path))[
                                 0] for video_path in audios_sliced_list])

    return total_sliced_data


def slice_single_tuple(
    data_dir: str,
    video_path: str,
    music_path: str,
    motion_path: str,
    length_frames: int = 243,
    target_fps: int = 30,
    original_fps: int = None,
    skip_if_exists: bool = False,
):
    """
    Slices a single (video, pose) pair into aligned 243-frame windows.
    """
    assert os.path.exists(video_path) and os.path.isfile(
        video_path), f"Video path ${video_path} not found"
    assert os.path.exists(music_path) and os.path.isfile(
        music_path), f"Audio path ${music_path} not found"
    assert os.path.exists(motion_path) and os.path.isfile(
        motion_path), f"Motion path ${motion_path} not found"
    audio_slice_out_dir = os.path.join(data_dir, "wavs_sliced")
    os.makedirs(audio_slice_out_dir, exist_ok=True)
    video_slice_out_dir = os.path.join(data_dir, "video_sliced")
    os.makedirs(video_slice_out_dir, exist_ok=True)
    motion_slice_out_dir = os.path.join(data_dir, "motions_sliced")
    os.makedirs(motion_slice_out_dir, exist_ok=True)

    if skip_if_exists:
        tqdm.write("Collecting existing slices...")
        basename = os.path.splitext(os.path.basename(video_path))[0]
        audio_existing = find_existing_slices(
            audio_slice_out_dir,  basename, "wav")
        video_existing = find_existing_slices(
            video_slice_out_dir,  basename, "mp4")
        motion_existing = find_existing_slices(
            motion_slice_out_dir, basename, "pkl")

        if video_existing or motion_existing or audio_existing:
            if (audio_existing and video_existing and motion_existing and
                    len(audio_existing) == len(video_existing) == len(motion_existing)):
                tqdm.write(
                    f"Enough existing slices found for {basename}, skipping slicing.")
                return audio_existing, video_existing, motion_existing

            # Mismatch → log và slice lại (KHÔNG return)
            tqdm.write(
                f"[WARN] slice mismatch {basename}, re-slicing: "
                f"audio={len(audio_existing)}, "
                f"video={len(video_existing)}, "
                f"motion={len(motion_existing)}"
            )

    vr = VideoReader(video_path, ctx=cpu(0))

    if not original_fps:
        original_fps = vr.get_avg_fps()
    # print(f"Video {video_path} has fps {original_fps}")

    # slice audio
    audio_slices = slice_audio(
        audio_path=music_path,
        output_dir=audio_slice_out_dir,
        n_frames=length_frames,
        fps=target_fps,
        overlap=1.0,
    )
    assert len(audio_slices) > 0, f"No audio slices created for {music_path}"
    # print(f"Audio slices: {audio_slices}")

    # slice video
    video_slices = slice_video(
        video_path=video_path,
        output_dir=video_slice_out_dir,
        n_frames=length_frames,
        target_fps=target_fps,
        overlap=1,
    )
    assert len(video_slices) > 0, f"No video slices created for {video_path}"

    # slice motion
    motion_slices = slice_motion(
        motion_path=motion_path,
        output_dir=motion_slice_out_dir,
        n_frames=length_frames,
        target_fps=target_fps,
    )
    assert len(
        motion_slices) > 0, f"No motion slices created for {motion_path}"
    # print(f"Motion slices: {motion_slices}")

    assert len(audio_slices) == len(video_slices) == len(
        motion_slices), (len(audio_slices), len(video_slices), len(motion_slices))
    return audio_slices, video_slices, motion_slices


def find_existing_slices(out_dir: str, basename: str, ext: str):
    """
    Return sorted slice names WITHOUT extension.
    Example: ['xxx_slice0', 'xxx_slice1']
    """
    if not os.path.isdir(out_dir):
        return []

    pattern = re.compile(
        rf"^{re.escape(basename)}_slice(\d+)\.{re.escape(ext)}$")
    slices = []

    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m:
            slices.append((int(m.group(1)), os.path.splitext(f)[0]))

    return [name for _, name in sorted(slices)]
