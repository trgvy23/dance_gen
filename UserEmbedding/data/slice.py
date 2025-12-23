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

from extraction_features.video_masks_features import get_video_masks_features
from extraction_features.video_features import extract_video_features

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
    class_to_idx = {cls: idx for (idx, cls) in enumerate(
        weights.meta["categories"])}
    person_idx = class_to_idx["person"]  # human class

    return model, preprocess, person_idx


def slice_audio(
    audio_file,
    length_frames,
    out_dir,
    original_fps=ORIGINAL_FPS,
):
    """
    Slice audio into chunks based on a given number of *frames*.

    - stride_frames: how many frames to move between slices
    - length_frames: how many frames per slice (e.g. 243)
    - fps: frame rate of your motion / features (e.g. 60 for 60 fps)
    """
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    T = len(audio)

    # how many audio samples correspond to 1 frame
    samples_per_frame = int(round(sr / original_fps))

    window = length_frames * samples_per_frame

    start_idx = 0
    idx = 0
    expected_length = -1

    print(f"Slicing audio {audio_file} with {T} samples...")
    # while start_idx <= len(audio) - window:
    while start_idx < T:
        print(f"Slicing audio {idx} from sample {start_idx}...")
        audio_slice = audio[start_idx: min(start_idx + window, T)]
        # print(f"Type of audio slice: {type(audio_slice)}")
        # print(f"  audio slice shape: {len(audio_slice)}")
        # print(f"  values: {audio_slice[0]}")
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        if expected_length == -1:
            expected_length = len(audio_slice)
        elif len(audio_slice) < expected_length:
            pad_len = expected_length - len(audio_slice)
            padding = np.zeros(pad_len, dtype=audio_slice.dtype)
            audio_slice = np.concatenate([audio_slice, padding], axis=0)
            sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
            # print(f"Padded audio slice {idx} with {pad_len} zeros.")
            # print(f"  new shape: {len(audio_slice)}")
            # print(f"  new values: {audio_slice[-1]}")
        start_idx += window
        idx += 1

    print(expected_length)
    return idx


def slice_video_masks(
    video_path: str,
    length_frames: int,
    step: int,
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

    # Reader for masks; use original resolution unless mask_reader_size is provided
    if mask_reader_size is None:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        h, w = mask_reader_size
        vr = VideoReader(video_path, width=w, height=h, ctx=cpu(0))

    T = len(vr)
    start = 0
    idx = 0
    basename = os.path.splitext(os.path.basename(video_path))[0]
    slice_length = step * length_frames

    print(f"Slicing video masks {video_path} with {T} frames...")
    with torch.no_grad():
        print(f"Slicing video mask {idx} from frame {start}...")
        inds = list(range(T))
        frames = vr.get_batch(inds).asnumpy()  # [T,H,W,3] uint8
        T_slice = frames.shape[0]
        masks = get_video_masks_features(  # [T,64,64]
            frames,
            T_slice,
            seg_preprocess,
            person_idx,
            seg_batch,
            seg_model,
            mask_latent_size,
            device,
        )
        if masks.shape != torch.Size([length_frames, 64, 64]):
            cur = masks.shape[0]
            pad_len = length_frames - cur
            padding = torch.zeros(
                pad_len, 64, 64, device=masks.device, dtype=masks.dtype
            )
            masks = torch.cat([masks, padding], dim=0)
        np.save(
            os.path.join(mask_output_dir, f"{basename}_slice{idx}.npy"),
            masks.cpu().numpy().astype(np.float32),
        )

        start += slice_length
        idx += 1

    return idx


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
        number of segments created
    """
    assert 0 < overlap <= 1.0, "overlap must be in (0, 1]"

    os.makedirs(output_dir, exist_ok=True)

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
    print(f"\tSlicing video {video_path}")
    for start in range(0, total_frames, segment_stride):
        print(f"\t\tSlice from frame {start} to {start + n_frames}")
        segment = frames[start: start + n_frames]
        if len(segment) == 0:
            break

        # Pad last segment if needed
        if len(segment) < n_frames:
            print(f"Missing frame at segment {start} - {start + n_frames}")
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
    slice_videos_list: list,
    vid_feature_output_dir: str,
    videoprism_model,
    use_bfloat16: bool = True,
) -> list:
    fprop_dtype = jnp.bfloat16 if use_bfloat16 else None
    flax_model = vp.get_model(videoprism_model, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(videoprism_model)

    feature_output_dir = []
    for video_path in slice_videos_list:
        vr = VideoReader(video_path, width=VIDEO_WIDTH,
                         height=VIDEO_HEIGHT, ctx=cpu(0))
        T = len(vr)
        inds = list(range(T))
        batch = vr.get_batch(inds).asnumpy()
        embeddings = extract_video_features(
            batch, fprop_dtype, flax_model, loaded_state)
        output_file = os.path.join(
            vid_feature_output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npy")
        feature_output_dir.append(output_file)
        print(f"\t\tSaving video features to {output_file}")
        np.save(output_file, embeddings)

    return feature_output_dir

# TODO: add get video mask for each video segment


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
    data_stride: int,
    length_frames: int = 243,
    fps: int = ORIGINAL_FPS,
):
    with open(motion_path, "rb") as motion_pkl:
        motion_data = pickle.load(motion_pkl)
    T = len(motion_data["smpl_poses"])

    # print(f"Slicing motion {motion_path}...")
    # step = ORIGINAL_FPS // fps if fps is not None else 1

    start = 0
    idx = 0
    slice_length = data_stride * length_frames
    basename = os.path.splitext(os.path.basename(motion_path))[0]

    print(f"Slicing motion {motion_path} with {T} frames...")
    keys = motion_data.keys()

    while start < T:
        sliced_motion = dict.fromkeys(keys)
        # print(f"Slicing motion {idx} from frame {start}...")
        for key in motion_data.keys():
            if key == "smpl_loss":
                sliced_motion[key] = 0.0
            elif key == "smpl_scaling":
                sliced_motion[key] = motion_data[key]
            else:
                # print(
                #     type(start), start,
                #     type(slice_length), slice_length,
                #     type(T), T,
                #     type(data_stride), data_stride
                # )

                sliced_motion[key] = motion_data[key][
                    start: min(start + slice_length, T): data_stride
                ]
                # print(f"Length of sliced motion at key {key}: {len(sliced_motion[key])}")
                if len(sliced_motion[key]) < length_frames:
                    cur = len(sliced_motion[key])
                    pad_len = length_frames - cur
                    if isinstance(motion_data[key], np.ndarray):
                        padding = np.zeros(
                            (pad_len,) + motion_data[key].shape[1:],
                            dtype=motion_data[key].dtype,
                        )
                        sliced_motion[key] = np.vstack(
                            [sliced_motion[key], padding])
                    else:
                        padding = [0] * pad_len
                        sliced_motion[key].extend(padding)
                    # print(f"Length of sliced motion at key {key} after padding: {len(sliced_motion[key])}")
        # print(f"Sliced motion: {sliced_motion}")
        with open(
            os.path.join(output_dir, f"{basename}_slice{idx}.pkl"), "wb"
        ) as write_file:
            pickle.dump(sliced_motion, write_file)
        start += slice_length
        idx += 1

    # print(f"Motion type: {type(motion_data)}")
    # for key in motion_data.keys():
    #     if hasattr(motion_data[key], "shape"):
    #         print(f"  Key: {key}, type: {motion_data[key]}, shape: {motion_data[key].shape}")
    #     else:
    #         print(f"  Key: {key}, type: {motion_data[key]}")

    # print(f"Motion content: {motion_data}")
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
    vid_feature_out = video_dir + "_embedding_sliced"
    vid_feature_out = vid_feature_out.replace("vyttt/", "vyttt/catb/")
    vid_mask_out = video_dir + "_mask_sliced"
    vid_mask_out = vid_mask_out.replace("vyttt/", "vyttt/catb/")
    pose_est_out = pose_estimation_dir + "_sliced"
    pose_est_out = pose_est_out.replace("vyttt/", "vyttt/catb/")

    ensure_dir(vid_feature_out)
    ensure_dir(pose_est_out)
    ensure_dir(vid_mask_out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # seg_model, seg_preprocess, person_idx = build_segmentation_model(device=device)
    seg_model, seg_preprocess, person_idx = build_segmentation_model(
        device="cpu")

    video_files = sorted(glob.glob(f"{video_dir}/*.mp4"))
    pose_est_files = sorted(glob.glob(f"{pose_estimation_dir}/*.json"))
    assert len(video_files) == len(pose_est_files), (
        "Mismatch between video and pose estimation file counts"
    )

    data_stride = ORIGINAL_FPS // fps if fps is not None else 1
    print(f"Slicing dataset with data stride {data_stride}...")

    for video_path, pose_path in tqdm(zip(video_files, pose_est_files), desc="Slicing"):
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        pose_basename = os.path.splitext(os.path.basename(pose_path))[0]
        # vr = VideoReader(video_path, ctx=cpu(0))
        # fps = int(vr.get_avg_fps())
        # print(f"Video {video_path} has fps {fps}")
        fps = 60
        data_stride = ORIGINAL_FPS // fps if fps is not None else 1

        assert video_basename == pose_basename, str(
            (video_basename, pose_basename))

        video_slices = slice_video(
            video_path=video_path,
            length_frames=length_frames,
            step=data_stride,
            feature_output_dir=vid_feature_out,
            videoprism_model=VIDEOPRISM_MODEL_NAME,
        )
        print(f"Video slices: {video_slices}")
        pose_slices = slice_motion_estimation(
            pose_path=pose_path,
            length_frames=length_frames,
            step=data_stride,
            output_dir=pose_est_out,
        )
        print(f"Pose slices: {pose_slices}")
        mask_slices = slice_video_masks(
            video_path=video_path,
            length_frames=length_frames,
            step=data_stride,
            mask_output_dir=vid_mask_out,
            seg_model=seg_model,
            seg_preprocess=seg_preprocess,
            person_idx=person_idx,
            seg_batch=8,
            mask_latent_size=(64, 64),
            device=device,
        )
        print(f"Mask slices: {mask_slices}")
        # make sure the slices line up
        assert video_slices == pose_slices == mask_slices, (
            video_path,
            pose_path,
            video_slices,
            pose_slices,
            mask_slices,
        )


def slice_single_pair(
    audio_path,
    video_path,
    motion_feature_path,
    motion_path,
    length_frames: int = 243,
    target_fps: int = 30,  # TODO: fix fps feeding to model
    original_fps: int = None,
):
    """
    Slices a single (video, pose) pair into aligned 243-frame windows.
    """
    OUTPUT_DIR = "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp"
    video_slice_out_dir = f"{OUTPUT_DIR}/video_sliced/"
    ensure_dir(video_slice_out_dir)
    vid_feature_out_dir = f"{OUTPUT_DIR}/video_features_sliced/"
    ensure_dir(vid_feature_out_dir)
    # print(vid_feature_out, pose_est_out, pose_est_out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # seg_model, seg_preprocess, person_idx = build_segmentation_model(device=device)
    seg_model, seg_preprocess, person_idx = build_segmentation_model(
        device="cpu")

    vr = VideoReader(video_path, ctx=cpu(0))
    # methods = [
    #    name for name in dir(vr)
    #    if callable(getattr(vr, name)) and not name.startswith('__')
    # ]
    # print(methods)

    if not original_fps:
        original_fps = vr.get_avg_fps()
    data_stride = int(original_fps // target_fps)
    print(f"Video {video_path} has fps {target_fps}")
    print(f"Slicing dataset with data stride {data_stride}...")

    # audio_slices = slice_audio(
    #     audio_file=audio_path,
    #     stride_frames=data_stride,
    #     length_frames=length_frames,
    #     out_dir=audio_slice_out,
    #     fps=ORIGINAL_FPS,
    # )
    # print(f"Audio slices: {audio_slices}")

    # motion_slices = slice_motion(
    #     motion_path=motion_path,
    #     output_dir=motion_out,
    #     data_stride=data_stride,
    #     length_frames=length_frames,
    #     fps=fps_feed_to_model,
    # )
    # print(f"Motion slices: {motion_slices}")

    # video_slices = slice_video(
    #     video_path=video_path,
    #     length_frames=length_frames,
    #     step=data_stride,
    #     feature_output_dir=vid_feature_out,
    #     videoprism_model=VIDEOPRISM_MODEL_NAME,
    # )
    video_slices = slice_video(
        video_path=video_path,
        output_dir=video_slice_out_dir,
        n_frames=length_frames,
        target_fps=30,
        overlap=1,
    )
    print(f"Video slices: {video_slices}")
    video_features = extract_sliced_videos_features(
        slice_videos_list=video_slices,
        vid_feature_output_dir=vid_feature_out_dir,
        videoprism_model=VIDEOPRISM_MODEL_NAME,
    )
    print(f"Video features: {video_features}")
    # pose_slices = slice_motion_feature(
    #     pose_path=motion_feature_path,
    #     length_frames=length_frames,
    #     step=data_stride,
    #     output_dir=pose_est_out,
    # )
    # print(f"Pose slices: {pose_slices}")
    # mask_slices = slice_video_masks(
    #     video_path=video_path,
    #     length_frames=length_frames,
    #     step=data_stride,
    #     mask_output_dir=vid_mask_out,
    #     seg_model=seg_model,
    #     seg_preprocess=seg_preprocess,
    #     person_idx=person_idx,
    #     seg_batch=2,
    #     mask_latent_size=(64, 64),
    #     # device=device,
    #     device="cpu",
    # )
    # print(f"Mask slices: {mask_slices}")
    # # make sure the slices line up
    # assert audio_slices == motion_slices == video_slices == pose_slices == mask_slices, \
    #     (audio_slices, motion_slices, video_slices, pose_slices, mask_slices)


if __name__ == "__main__":
    slice_single_pair(
        audio_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/wavs/gBR_sBM_c01_d04_mBR0_ch04.wav",
        video_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video/gBR_sBM_c01_d04_mBR0_ch04.mp4",
        motion_feature_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation/gBR_sBM_c01_d04_mBR0_ch04.json",
        motion_path="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/motions/gBR_sBM_c01_d04_mBR0_ch04.pkl",
        length_frames=243,
    )
    # slice_dataset(
    #     video_dir="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video",
    #     pose_estimation_dir="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation",
    #     length_frames=243,
    #     # fps=30,
    # )
