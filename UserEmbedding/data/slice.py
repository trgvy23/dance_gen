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

from extraction_features.video_mask_features import get_video_masks_features
from extraction_features.video_features import extract_video_features
from extraction_features.audio_baseline_features import extract_folder
from extraction_features import alphapose_features
from extraction_features import motionbert_features

ORIGINAL_FPS = 60  # original fps of videos in dataset
VIDEO_WIDTH = 288
VIDEO_HEIGHT = 288
VIDEOPRISM_MODEL_NAME = "videoprism_public_v1_base"


def ensure_dir(p: str) -> None:
    if os.path.exists(p):
        # Xóa toàn bộ nội dung bên trong directory
        for name in os.listdir(p):
            path = os.path.join(p, name)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
    else:
        os.makedirs(p, exist_ok=True)


def build_segmentation_model(device: str = "cuda"):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
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

    print(f"Slicing audio {audio_path}...")
    print(f"\tSample rate: {sr}, FPS: {fps}")
    print(f"\tWindow: {n_frames} frames -> {samples_per_window} samples")
    print(f"\tStep: {overlap} -> {step_samples} samples")

    # Iterate while the start of the slice is within the limit
    while start_idx < total_samples:
        print(
            f"\tSlice {idx} from {start_idx} to {min(start_idx + samples_per_window, total_samples)}"
        )

        # Determine strict end if we want to mimic video 'break' behavior?
        # In video: break if start + n_frames >= total_frames
        # That means the last slice must be 'started' before the end, but if it goes over, it's padded.
        # But if the next start is >= total, we stop.
        # Here: start_idx < limit_samples handles that.

        end_idx = min(start_idx + samples_per_window, total_samples)
        audio_slice = audio[start_idx:end_idx]

        # Pad with zeros if the slice is shorter than expected (last slice)
        if len(audio_slice) < samples_per_window:
            print(f"\t\tMissing audio at segment {start_idx} - {end_idx}")
            pad_len = samples_per_window - len(audio_slice)
            audio_slice = np.pad(audio_slice, (0, pad_len), mode="constant")

        out_path = os.path.join(output_dir, f"{file_name}_slice{idx}.wav")
        sf.write(out_path, audio_slice, sr)

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
    print(f"Slicing video {video_path}")
    for start in range(0, total_frames, segment_stride):
        print(f"\tSlice from frame {start} to {start + n_frames}")
        segment = frames[start : start + n_frames]
        if len(segment) == 0:
            break

        # Pad last segment if needed
        if len(segment) < n_frames:
            print(f"\t\tMissing frame at segment {start} - {start + n_frames}")
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
        segment_count += 1

        # Stop if this was the padded last segment
        if start + n_frames >= total_frames:
            break

    return slice_videos_list


def extract_sliced_videos_features(
    sliced_videos_list: list,
    vid_feature_output_dir: str,
    videoprism_model,
    use_bfloat16: bool = True,
) -> list:
    ensure_dir(vid_feature_output_dir)
    fprop_dtype = jnp.bfloat16 if use_bfloat16 else None
    flax_model = vp.get_model(videoprism_model, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(videoprism_model)

    feature_output_dir = []
    for video_path in sliced_videos_list:
        vr = VideoReader(video_path, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, ctx=cpu(0))
        T = len(vr)
        inds = list(range(T))
        batch = vr.get_batch(inds).asnumpy()
        embeddings = extract_video_features(
            batch, fprop_dtype, flax_model, loaded_state
        )
        output_file = os.path.join(
            vid_feature_output_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}.npy",
        )
        feature_output_dir.append(output_file)
        print(f"\t\tSaving video features to {output_file}")
        np.save(output_file, embeddings)

    return feature_output_dir


def extract_sliced_videos_masks(
    sliced_videos_list: list,
    mask_output_dir: str,
    seg_model,
    seg_preprocess,
    person_idx: int,
    seg_batch: int = 16,
    mask_latent_size: Tuple[int, int] = (64, 64),
    device: str = "cpu",
    mask_reader_size: Optional[Tuple[int, int]] = (
        512,
        512,
    ),  # e.g. (720, 1280) or None for original
) -> int:
    """
    Runs segmentation in mini-batches for each slice and saves [T, h_latent, w_latent] masks.
    """
    ensure_dir(mask_output_dir)
    video_masked_list = []
    with torch.no_grad():
        for video_path in sliced_videos_list:
            # Reader for masks; use original resolution unless mask_reader_size is provided
            if mask_reader_size is None:
                vr = VideoReader(video_path, ctx=cpu(0))
            else:
                h, w = mask_reader_size
                vr = VideoReader(video_path, width=w, height=h, ctx=cpu(0))
            T = len(vr)
            inds = list(range(T))
            frames = vr.get_batch(inds).asnumpy()  # [T,H,W,3] uint8
            T_slice = frames.shape[0]
            masks = get_video_masks_features(  # [T,64,64]
                frames,
                T_slice,
                seg_model,
                seg_preprocess,
                person_idx,
                seg_batch,
                mask_latent_size,
                device,
            )
            vids_masked_out_file_path = os.path.join(
                mask_output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}.npy",
            )
            np.save(
                vids_masked_out_file_path,
                masks.cpu().numpy().astype(np.float32),
            )
            video_masked_list.append(vids_masked_out_file_path)
            print(f"Saving video masks to {vids_masked_out_file_path}")

    return video_masked_list


def slice_motion_estimation(
    pose_path: str, length_frames: int, step: int, output_dir: str
) -> int:
    with open(pose_path, "r") as read_file:
        pose_ests = json.load(read_file)

    T = len(pose_ests)

    print(f"Slicing motion estimation {pose_path} with {T} frames...")
    start = 0
    idx = 0
    basename = os.path.splitext(os.path.basename(pose_path))[0]

    slice_length = step * length_frames
    padding_keypoints = [0] * 78
    padding_keypoints = {
        "keypoints": padding_keypoints,
    }
    while start < T:
        print(f"Slicing motion estimation {idx} from frame {start}...")
        slice_ests = pose_ests[start : min(start + slice_length, T) : step]
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

    print(f"Slicing motion {motion_path}")
    print(f"  Total frames: {T}")
    print(f"  data_stride: {data_stride}")
    print(f"  segment_stride: {segment_stride}")

    slice_paths = []
    segment_count = 0

    for start in range(0, T, segment_stride * data_stride):
        print(f"\tSlice from frame {start} to {start + slice_length}")

        sliced_motion = {}

        for key in keys:
            if key == "smpl_loss":
                sliced_motion[key] = 0.0
            elif key == "smpl_scaling":
                sliced_motion[key] = motion_data[key]
            else:
                chunk = motion_data[key][
                    start : min(start + slice_length, T) : data_stride
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

        slice_paths.append(out_path)
        segment_count += 1

        # Stop after padded last segment (same logic as video)
        if start + slice_length >= T:
            break

    return slice_paths



def extract_feat_using_alphapose(
    sliced_vid_list: list,
    raw_alphapose_out_dir: list,
    final_alphapose_out_dir: str,
    conda_env: str = "alphapose",
) -> list:
    """
    Run AlphaPose inference on sliced videos.

    Args:
        sliced_vid_list: list of sliced .mp4 paths
        raw_alphapose_out_dir: directory to save raw AlphaPose outputs
        final_alphapose_out_dir: directory to save final AlphaPose outputs
        conda_env: conda environment name for AlphaPose

    Returns:
        list of output file paths
    """
    ensure_dir(raw_alphapose_out_dir)
    ensure_dir(final_alphapose_out_dir)
    env = os.environ.copy()
    cuda_device = env["CUDA_VISIBLE_DEVICES"]
    alphapose_feats_list = alphapose_features.run_alphapose(
        sliced_vid_list,
        raw_alphapose_out_dir,
        final_alphapose_out_dir,
        cuda_device,
        conda_env,
    )
    return alphapose_feats_list


#TODO: run motionbert
def extract_feat_using_motionbert(
    sliced_vid_list: list,
    sliced_alphapose_list: list,
    sliced_motionbert_dir: str,
    conda_env: str = "motionbert",
) -> list:
    """
    Run MotionBERT inference on sliced videos + AlphaPose jsons.

    Args:
        sliced_vid_list: list of sliced .mp4 paths
        sliced_alphapose_list: list of corresponding alphapose .json paths
        sliced_motionbert_dir: directory to save MotionBERT outputs
        conda_env: conda environment name for MotionBERT

    Returns:
        list of output file paths
    """
    ensure_dir(sliced_motionbert_dir)
    # make sure video and alphapose have the same basename
    for v, j in zip(sliced_vid_list, sliced_alphapose_list):
        print(f"video file name: {os.path.splitext(os.path.basename(v))[0]}, alphapose file name: {os.path.splitext(os.path.basename(j))[0]}")
        assert os.path.splitext(os.path.basename(v))[0] == \
               os.path.splitext(os.path.basename(j))[0]
    env = os.environ.copy()
    cuda_device = env["CUDA_VISIBLE_DEVICES"]
    motionbert_feats_list = motionbert_features.run_motionbert(
        sliced_vid_list,
        sliced_alphapose_list,
        sliced_motionbert_dir,
        cuda_device,
        conda_env,
    )
    return motionbert_feats_list


# def slice_dataset(
#     video_dir,
#     pose_estimation_dir,
#     length_frames: int = 243,
#     fps: int = None,
# ):
#     """
#     Slices each (video, pose) pair into aligned 243-frame windows.
#     """
#     vid_feature_out = video_dir + "_embedding_sliced"
#     vid_feature_out = vid_feature_out.replace("vyttt/", "vyttt/catb/")
#     vid_mask_out = video_dir + "_mask_sliced"
#     vid_mask_out = vid_mask_out.replace("vyttt/", "vyttt/catb/")
#     pose_est_out = pose_estimation_dir + "_sliced"
#     pose_est_out = pose_est_out.replace("vyttt/", "vyttt/catb/")

#     ensure_dir(vid_feature_out)
#     ensure_dir(pose_est_out)
#     ensure_dir(vid_mask_out)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # seg_model, seg_preprocess, person_idx = build_segmentation_model(device=device)
#     seg_model, seg_preprocess, person_idx = build_segmentation_model(
#         device="cpu")

#     video_files = sorted(glob.glob(f"{video_dir}/*.mp4"))
#     pose_est_files = sorted(glob.glob(f"{pose_estimation_dir}/*.json"))
#     assert len(video_files) == len(pose_est_files), (
#         "Mismatch between video and pose estimation file counts"
#     )

#     data_stride = ORIGINAL_FPS // fps if fps is not None else 1
#     print(f"Slicing dataset with data stride {data_stride}...")

#     for video_path, pose_path in tqdm(zip(video_files, pose_est_files), desc="Slicing"):
#         video_basename = os.path.splitext(os.path.basename(video_path))[0]
#         pose_basename = os.path.splitext(os.path.basename(pose_path))[0]
#         # vr = VideoReader(video_path, ctx=cpu(0))
#         # fps = int(vr.get_avg_fps())
#         # print(f"Video {video_path} has fps {fps}")
#         fps = 60
#         data_stride = ORIGINAL_FPS // fps if fps is not None else 1

#         assert video_basename == pose_basename, str(
#             (video_basename, pose_basename))

#         video_slices = slice_video(
#             video_path=video_path,
#             length_frames=length_frames,
#             step=data_stride,
#             feature_output_dir=vid_feature_out,
#             videoprism_model=VIDEOPRISM_MODEL_NAME,
#         )
#         print(f"Video slices: {video_slices}")
#         pose_slices = slice_motion_estimation(
#             pose_path=pose_path,
#             length_frames=length_frames,
#             step=data_stride,
#             output_dir=pose_est_out,
#         )
#         print(f"Pose slices: {pose_slices}")
#         mask_slices = slice_video_masks(
#             video_path=video_path,
#             length_frames=length_frames,
#             step=data_stride,
#             mask_output_dir=vid_mask_out,
#             seg_model=seg_model,
#             seg_preprocess=seg_preprocess,
#             person_idx=person_idx,
#             seg_batch=8,
#             mask_latent_size=(64, 64),
#             device=device,
#         )
#         print(f"Mask slices: {mask_slices}")
#         # make sure the slices line up
#         assert video_slices == pose_slices == mask_slices, (
#             video_path,
#             pose_path,
#             video_slices,
#             pose_slices,
#             mask_slices,
#         )


def slice_single_tuple(
    audio_path,
    video_path,
    motion_path,
    length_frames: int = 243,
    target_fps: int = 30,
    original_fps: int = None,
):
    """
    Slices a single (video, pose) pair into aligned 243-frame windows.
    """
    OUTPUT_DIR = (
        "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp"
    )
    audio_slice_out_dir = f"{OUTPUT_DIR}/audio_sliced/"
    baseline_feat_sliced_audio = f"{OUTPUT_DIR}/baseline_feats"
    jukebox_feat_sliced_audio = f"{OUTPUT_DIR}/jukebox_feats"
    video_slice_out_dir = f"{OUTPUT_DIR}/video_sliced/"
    vid_feature_out_dir = f"{OUTPUT_DIR}/video_features_sliced/"
    vid_mask_out_dir = f"{OUTPUT_DIR}/video_mask_sliced/"
    motion_slice_out_dir = f"{OUTPUT_DIR}/motion_sliced/"
    raw_alphapose_out_dir = f"{OUTPUT_DIR}/alphapose_raw_sliced"
    final_alphapose_out_dir = f"{OUTPUT_DIR}/alphapose_sliced"
    motionbert_out_dir = f"{OUTPUT_DIR}/motionbert_sliced"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_preprocess, person_idx = build_segmentation_model(device="cpu")

    vr = VideoReader(video_path, ctx=cpu(0))

    if not original_fps:
        original_fps = vr.get_avg_fps()
    print(f"Video {video_path} has fps {original_fps}")

    # # slice audio
    # audio_slices = slice_audio(
    #     audio_path=audio_path,
    #     output_dir=audio_slice_out_dir,
    #     n_frames=length_frames,
    #     fps=target_fps,
    #     overlap=1.0,
    # )
    # print(f"Audio slices: {audio_slices}")

    # baseline_feat_sliced_audio = extract_baseline_feat_sliced_audio(
    #     audio_slice_out_dir, baseline_feat_sliced_audio
    # )
    # print(f"Baseline feat sliced audio: {baseline_feat_sliced_audio}")

    # jukebox_feat_sliced_audio = extract_jukebox_feat_sliced_audio(
    #     audio_slice_out_dir, jukebox_feat_sliced_audio
    # )
    # print(f"Jukebox feat sliced audio: {jukebox_feat_sliced_audio}")

    # slice video
    video_slices = slice_video(
       video_path=video_path,
       output_dir=video_slice_out_dir,
       n_frames=length_frames,
       target_fps=target_fps,
       overlap=1,
    )
    print(f"Video slices: {video_slices}")

    video_features = extract_sliced_videos_features(
        sliced_videos_list=video_slices,
        vid_feature_output_dir=vid_feature_out_dir,
        videoprism_model=VIDEOPRISM_MODEL_NAME,
    )
    print(f"Video features: {video_features}")

    video_masks = extract_sliced_videos_masks(
        sliced_videos_list=video_slices,
        mask_output_dir=vid_mask_out_dir,
        seg_model=seg_model,
        seg_preprocess=seg_preprocess,
        person_idx=person_idx,
        seg_batch=2,
        mask_latent_size=(64, 64),
        # device=device,
        device="cpu",
    )
    print(f"Video masks: {video_masks}")

    alphapose_feat_list = extract_feat_using_alphapose(
        video_slices,
        raw_alphapose_out_dir,
        final_alphapose_out_dir
    )
    print(f"Alphapose features: {alphapose_feat_list}")

    motionbert_feat_list = extract_feat_using_motionbert(
        video_slices,
        alphapose_feat_list,
        motionbert_out_dir,
    )
    print(f"MotionBERT features: {motionbert_feat_list}")

    # # slice motion
    # motion_slices = slice_motion(
    #     motion_path=motion_path,
    #     output_dir=motion_slice_out_dir,
    #     n_frames=length_frames,
    #     target_fps=target_fps,
    # )
    # print(f"Motion slices: {motion_slices}")

    # pose_slices = slice_motion_feature(
    #     pose_path=motion_feature_path,
    #     length_frames=length_frames,
    #     step=data_stride,
    #     output_dir=pose_est_out,
    # )
    # print(f"Pose slices: {pose_slices}")
    # # make sure the slices line up
    # assert audio_slices == motion_slices == video_slices == pose_slices == mask_slices, \
    #     (audio_slices, motion_slices, video_slices, pose_slices, mask_slices)


if __name__ == "__main__":
    slice_single_tuple(
        audio_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/wavs/gBR_sBM_c01_d04_mBR0_ch04.wav",
        video_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video/gBR_sBM_c01_d04_mBR0_ch04.mp4",
        # motion_feature_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation/gBR_sBM_c01_d04_mBR0_ch04.json",
        motion_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/motions/gBR_sBM_c01_d04_mBR0_ch04.pkl",
        length_frames=243,
    )
    # slice_dataset(
    #     video_dir="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video",
    #     pose_estimation_dir="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation",
    #     length_frames=243,
    #     # fps=30,
    # )
