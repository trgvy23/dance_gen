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


from decord import VideoReader, cpu
import jax
import jax.numpy as jnp
from videoprism import models as vp
import torch.nn.functional as F

from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)

ORIGINAL_FPS = 60  # original fps of videos in dataset
VIDEO_WIDTH = 288
VIDEO_HEIGHT = 288
VIDEOPRISM_MODEL_NAME = 'videoprism_public_v1_base'

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
    
def build_segmentation_model(device: str = "cuda"):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    class_to_idx = {
        cls: idx for (idx, cls) in enumerate(weights.meta["categories"])
    }
    person_idx = class_to_idx["person"]   # human class

    return model, preprocess, person_idx

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
    device: str = "cuda",
    mask_reader_size: Optional[Tuple[int, int]] = (512, 512),  # e.g. (720, 1280) or None for original
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

    with torch.no_grad():
        while start <= T - step * length_frames:
            inds = list(range(start, start + step * length_frames, step))
            frames = vr.get_batch(inds).asnumpy()              # [T,H,W,3] uint8
            T_slice = frames.shape[0]

            # Convert to torch [T,3,H,W]
            imgs = torch.from_numpy(frames).permute(0, 3, 1, 2)  # uint8
            # Apply preprocess per frame (keeps aspect ratio; short side ~520)
            inp_list = [seg_preprocess(img) for img in imgs]     # list of [3,H',W']
            inp = torch.stack(inp_list, dim=0).to(device)        # [T,3,H',W']

            all_masks = []
            for s in range(0, T_slice, seg_batch):
                chunk = inp[s : s + seg_batch]                   # [b,3,H',W']
                out = seg_model(chunk)["out"]                    # [b,C,H',W']
                probs = out.softmax(dim=1)                       # [b,C,H',W']
                person = probs[:, person_idx:person_idx+1]       # [b,1,H',W']

                # Downsample to latent 64x64 with nearest
                person_64 = F.interpolate(
                    person,
                    size=mask_latent_size,
                    mode="nearest",
                ).squeeze(1)                                     # [b,64,64]

                all_masks.append(person_64)

            masks = torch.cat(all_masks, dim=0)                  # [T,64,64]
            np.save(
                os.path.join(mask_output_dir, f"{basename}_slice{idx}.npy"),
                masks.cpu().numpy().astype(np.float32),
            )

            start += length_frames
            idx += 1

    return idx

def slice_video(
    video_path: str,
    length_frames: int,
    step: int,
    feature_output_dir: str,
    videoprism_model,
    use_bfloat16: bool = True,
) -> int:
    ensure_dir(feature_output_dir)
    
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
        
        # --- VideoPrism embedding extraction ---
        batch = mediapy.to_float01(batch)
        batch = torch.from_numpy(batch).unsqueeze(0).numpy()
        batch = jnp.asarray(batch, dtype=fprop_dtype or jnp.float32)
        
        embeddings, _ = flax_model.apply(loaded_state, batch, train=False)
        embeddings = embeddings.squeeze(0)  # [T, D]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        np.save(os.path.join(feature_output_dir, f"{basename}_slice{idx}.npy"), embeddings)
        
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
    vid_feature_out = video_dir + "_embedding_sliced"
    vid_mask_out = video_dir + "_mask_sliced"
    pose_est_out = pose_estimation_dir + "_sliced"

    ensure_dir(vid_feature_out)
    ensure_dir(pose_est_out)
    ensure_dir(vid_mask_out)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_preprocess, person_idx = build_segmentation_model(device=device)

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
            feature_output_dir=vid_feature_out,
            videoprism_model=VIDEOPRISM_MODEL_NAME,
        )
        pose_slices = slice_motion_estimation(
            pose_path=pose_path,
            length_frames=length_frames,
            step=data_stride,
            output_dir=pose_est_out,
        )
        mask_slices = slice_video_masks(
            video_path=video_path,
            length_frames=length_frames,
            step=data_stride,
            mask_output_dir=vid_mask_out,
            seg_model=seg_model,
            seg_preprocess=seg_preprocess,
            person_idx=person_idx,
            seg_batch=32, 
            mask_latent_size=(64, 64),
            device=device,
        )
        # make sure the slices line up
        assert video_slices == pose_slices == mask_slices, \
            (video_path, pose_path, video_slices, pose_slices, mask_slices)
