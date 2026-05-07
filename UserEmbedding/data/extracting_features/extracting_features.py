import time
import jax
import gc
from extracting_features import jukebox_features
from extracting_features import audio_baseline_features
from extracting_features.alphapose_features import run_alphapose
from extracting_features.motionbert_features import run_motionbert
from extracting_features.video_mask_features import get_video_masks_features
from extracting_features.video_features import extract_video_features
from tqdm import tqdm
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)
import torch
import numpy as np
import os
from decord import VideoReader, cpu
from videoprism import models as vp
import jax.numpy as jnp
from io_files import ensure_dir
from email.mime import base
import sys
from turtle import st
from unittest import skip
from venv import create
sys.path.append("..")


ORIGINAL_FPS = 60  # original fps of videos in dataset
VIDEO_WIDTH = 288
VIDEO_HEIGHT = 288
VIDEOPRISM_MODEL_NAME = "videoprism_public_v1_base"


def extract_sliced_videos_features(
    sliced_videos_list: list,
    vid_feature_output_dir: str,
    videoprism_model,
    use_bfloat16: bool = True,
    skip_if_exists: bool = False,
) -> list:
    for path in sliced_videos_list:
        assert os.path.exists(path), f"Sliced video path {path} not found"
    ensure_dir(vid_feature_output_dir, create_new=not skip_if_exists)
    fprop_dtype = jnp.bfloat16 if use_bfloat16 else None
    flax_model = vp.get_model(videoprism_model, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(videoprism_model)

    feature_output_dir = []
    pbar = tqdm(
        sliced_videos_list,
        desc="Extracting video features",
        unit="video",
    )
    for video_path in pbar:
        pbar.set_postfix(
            file=os.path.basename(video_path),
            refresh=False,
        )
        output_file = os.path.join(
            vid_feature_output_dir,
            f"{os.path.splitext(os.path.basename(video_path))[0]}.npy",
        )
        if skip_if_exists and os.path.exists(output_file):
            feature_output_dir.append(output_file)
            continue

        vr = VideoReader(video_path, width=VIDEO_WIDTH,
                         height=VIDEO_HEIGHT, ctx=cpu(0))
        T = len(vr)
        inds = list(range(T))
        batch = vr.get_batch(inds).asnumpy()
        embeddings = extract_video_features(
            batch, fprop_dtype, flax_model, loaded_state
        )
        jax.block_until_ready(embeddings)
        np.save(output_file, embeddings)
        feature_output_dir.append(output_file)
        # clear jax
        jax.clear_caches()
        # print(f"\t\tSaving video features to {output_file}")
        del embeddings, batch
        # python GC
        gc.collect()
        time.sleep(2)

    return feature_output_dir


def extract_sliced_videos_masks(
    sliced_videos_list: list,
    mask_output_dir: str,
    seg_model,
    seg_preprocess,
    person_idx: int,
    seg_batch: int = 16,
    mask_latent_size: tuple[int, int] = (64, 64),
    device: str = "cpu",
    mask_reader_size: tuple[int, int] = (
        512,
        512,
    ),  # e.g. (720, 1280) or None for original
    skip_if_exists: bool = False,
) -> int:
    """
    Runs segmentation in mini-batches for each slice and saves [T, h_latent, w_latent] masks.
    """
    for path in sliced_videos_list:
        assert os.path.exists(path), f"Sliced video path {path} not found"
    ensure_dir(mask_output_dir, create_new=not skip_if_exists)
    video_masked_list = []
    seg_model = seg_model.to(device)
    pbar = tqdm(sliced_videos_list,
                desc="Extracting video masks", unit="video")
    with torch.no_grad():
        for video_path in pbar:
            vids_masked_out_file_path = os.path.join(
                mask_output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}.npy",
            )
            pbar.set_postfix({"file": os.path.basename(video_path)})
            if skip_if_exists and os.path.exists(vids_masked_out_file_path):
                video_masked_list.append(vids_masked_out_file_path)
                continue

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
            masks = get_video_masks_features(
                frames=frames,
                T_slice=T_slice,
                seg_model=seg_model,
                seg_preprocess=seg_preprocess,
                person_idx=person_idx,
                seg_batch=seg_batch,
                mask_latent_size=mask_latent_size,
                device=device,
            )

            np.save(
                vids_masked_out_file_path,
                masks.cpu().numpy().astype(np.float32),
            )
            video_masked_list.append(vids_masked_out_file_path)
            # print(f"Saving video masks to {vids_masked_out_file_path}")

            del masks
            torch.cuda.empty_cache()

    return video_masked_list


def extract_feat_using_alphapose(
    sliced_vid_list: list,
    raw_alphapose_out_dir: list,
    final_alphapose_out_dir: str,
    conda_env: str = "alphapose",
    skip_if_exists: bool = False,
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
    for path in sliced_vid_list:
        assert os.path.exists(path), f"Sliced video path {path} not found"
    ensure_dir(raw_alphapose_out_dir, create_new=not skip_if_exists)
    ensure_dir(final_alphapose_out_dir, create_new=not skip_if_exists)
    env = os.environ.copy()
    cuda_device = env["CUDA_VISIBLE_DEVICES"]
    alphapose_feats_list = run_alphapose(
        sliced_vid_list,
        raw_alphapose_out_dir,
        final_alphapose_out_dir,
        cuda_device,
        conda_env,
        skip_if_exists=skip_if_exists,
    )
    return alphapose_feats_list


def extract_feat_using_motionbert(
    sliced_vid_list: list,
    sliced_alphapose_list: list,
    sliced_motionbert_dir: str,
    conda_env: str = "motionbert",
    skip_if_exists: bool = False,
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
    for path in sliced_vid_list:
        assert os.path.exists(path), f"Sliced video path {path} not found"
    ensure_dir(sliced_motionbert_dir, create_new=not skip_if_exists)
    # make sure video and alphapose have the same basename
    for v, j in zip(sliced_vid_list, sliced_alphapose_list):
        # print(
        #     f"video file name: {os.path.splitext(os.path.basename(v))[0]}, alphapose file name: {os.path.splitext(os.path.basename(j))[0]}")
        assert os.path.splitext(os.path.basename(v))[0] == \
            os.path.splitext(os.path.basename(j))[0]
    env = os.environ.copy()
    cuda_device = env["CUDA_VISIBLE_DEVICES"]
    motionbert_feats_list = run_motionbert(
        sliced_vid_list,
        sliced_alphapose_list,
        sliced_motionbert_dir,
        cuda_device,
        conda_env,
        skip_if_exists=skip_if_exists,
    )
    return motionbert_feats_list


def build_segmentation_model(device: str = "cuda"):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()
    class_to_idx = {cls: idx for (idx, cls) in enumerate(
        weights.meta["categories"])}
    person_idx = class_to_idx["person"]  # human class

    return model, preprocess, person_idx


def extract_baseline_feat_sliced_audio(
    sliced_audio_dir: str,
    baseline_feat_audio_dir: str,
    skip_if_exists: bool = False,
) -> list:
    ensure_dir(baseline_feat_audio_dir, create_new=not skip_if_exists)
    audio_baseline_features.extract_folder(
        sliced_audio_dir, baseline_feat_audio_dir, skip_completed=skip_if_exists)
    return [
        os.path.join(baseline_feat_audio_dir, f)
        for f in os.listdir(baseline_feat_audio_dir)
        if f.endswith(".npy")
    ]


def extract_jukebox_feat_sliced_audio(
    sliced_audio_dir: str,
    jukebox_feat_audio_dir: str,
    skip_if_exists: bool = False,
) -> list:
    ensure_dir(jukebox_feat_audio_dir, create_new=not skip_if_exists)
    jukebox_features.extract_folder(
        sliced_audio_dir, jukebox_feat_audio_dir, skip_completed=skip_if_exists)
    return [
        os.path.join(jukebox_feat_audio_dir, f)
        for f in os.listdir(jukebox_feat_audio_dir)
        if f.endswith(".npy")
    ]


def remove_data_in_directory(data_dir: str, data_list: list):
    data_list = set(data_list)

    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[0] in data_list:
                os.remove(os.path.join(root, f))


def extract_data_feats(
    data_dir: str,
    data_list: list,
    mask_width: int = 512,
    mask_height: int = 512,
    skip_if_exists: bool = False,
    alphapose_env: str = "alphapose",
    motionbert_env: str = "motionbert",
):
    audio_slice_out_dir = f"{data_dir}/wavs_sliced/"
    assert os.path.exists(
        audio_slice_out_dir), f"Audio slice dir not found: {audio_slice_out_dir}"
    video_slice_out_dir = f"{data_dir}/video_sliced/"
    assert os.path.exists(
        video_slice_out_dir), f"Video slice dir not found: {video_slice_out_dir}"
    motions_slice_out_dir = f"{data_dir}/motions_sliced/"
    assert os.path.exists(
        motions_slice_out_dir), f"Motions slice dir not found: {motions_slice_out_dir}"
    baseline_feat_sliced_audio_dir = f"{data_dir}/baseline_feats"
    ensure_dir(baseline_feat_sliced_audio_dir, create_new=not skip_if_exists)
    jukebox_feat_sliced_audio_dir = f"{data_dir}/jukebox_feats"
    ensure_dir(jukebox_feat_sliced_audio_dir, create_new=not skip_if_exists)
    vid_feature_out_dir = f"{data_dir}/video_features_sliced/"
    ensure_dir(vid_feature_out_dir, create_new=not skip_if_exists)
    vid_mask_out_dir = f"{data_dir}/video_mask_sliced/"
    ensure_dir(vid_mask_out_dir, create_new=not skip_if_exists)
    raw_alphapose_out_dir = f"{data_dir}/alphapose_raw_sliced"
    ensure_dir(raw_alphapose_out_dir, create_new=not skip_if_exists)
    final_alphapose_out_dir = f"{data_dir}/pose_estimation_sliced"
    ensure_dir(final_alphapose_out_dir, create_new=not skip_if_exists)
    motionbert_out_dir = f"{data_dir}/motionbert_sliced"
    ensure_dir(motionbert_out_dir, create_new=not skip_if_exists)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_preprocess, person_idx = build_segmentation_model(
        device=device)

    expected_num_samples = len(
        [f for f in os.listdir(audio_slice_out_dir) if os.path.isfile(os.path.join(audio_slice_out_dir, f))])

    print(
        f"Expected number of samples (based on sliced audio files): {expected_num_samples}")

    # extracting audio features
    baseline_feat_sliced_audio = extract_baseline_feat_sliced_audio(
        audio_slice_out_dir, baseline_feat_sliced_audio_dir, skip_if_exists=skip_if_exists
    )
    assert len(
        baseline_feat_sliced_audio) == expected_num_samples, f"Number of baseline audio features {len(baseline_feat_sliced_audio)} does not match number of data samples {expected_num_samples}"

    jukebox_feat_sliced_audio = extract_jukebox_feat_sliced_audio(
        audio_slice_out_dir, jukebox_feat_sliced_audio_dir, skip_if_exists=skip_if_exists
    )
    assert len(
        jukebox_feat_sliced_audio) == expected_num_samples, f"Number of jukebox audio features {len(jukebox_feat_sliced_audio)} does not match number of data samples {expected_num_samples}"

    # extracting video features
    sliced_vid_list = [os.path.abspath(os.path.join(video_slice_out_dir, f))
                       for f in os.listdir(video_slice_out_dir)
                       if os.path.isfile(os.path.join(video_slice_out_dir, f))]
    assert len(
        sliced_vid_list) == expected_num_samples, f"Number of sliced videos {len(sliced_vid_list)} does not match number of data samples {expected_num_samples}"
    # extract alphapose features
    alphapose_feat_list, remove_data = extract_feat_using_alphapose(
        sliced_vid_list,
        raw_alphapose_out_dir,
        final_alphapose_out_dir,
        conda_env=alphapose_env,
        skip_if_exists=skip_if_exists,
    )
    if len(remove_data) > 0:
        print(
            f"Warning: {len(remove_data)} videos failed AlphaPose processing and will be removed from dataset: {remove_data}")
        remove_data_in_directory(video_slice_out_dir, remove_data)
        remove_data_in_directory(audio_slice_out_dir, remove_data)
        remove_data_in_directory(motions_slice_out_dir, remove_data)
        remove_data_in_directory(baseline_feat_sliced_audio_dir, remove_data)
        remove_data_in_directory(jukebox_feat_sliced_audio_dir, remove_data)
        sliced_vid_list = [v for v in sliced_vid_list if os.path.splitext(
            os.path.basename(v))[0] not in remove_data]
        expected_num_samples -= len(remove_data)
    assert len(
        sliced_vid_list) == expected_num_samples, f"Number of sliced videos {len(sliced_vid_list)} does not match number of data samples {expected_num_samples}"
    # extract motionbert features
    motionbert_feat_list = extract_feat_using_motionbert(
        sliced_vid_list,
        alphapose_feat_list,
        motionbert_out_dir,
        conda_env=motionbert_env,
        skip_if_exists=skip_if_exists,
        # skip_if_exists=False,
    )
    assert len(
        motionbert_feat_list) == expected_num_samples, f"Number of MotionBERT features {len(motionbert_feat_list)} does not match number of data samples {expected_num_samples}"
    # extract video features
    vid_feats_list = extract_sliced_videos_features(
        sliced_vid_list,
        vid_feature_out_dir,
        VIDEOPRISM_MODEL_NAME,
        use_bfloat16=True,
        skip_if_exists=skip_if_exists
    )
    assert len(
        vid_feats_list) == expected_num_samples, f"Number of video features {len(vid_feats_list)} does not match number of data samples {expected_num_samples}"

    # extract video mask features
    vid_mask_list = extract_sliced_videos_masks(
        sliced_videos_list=sliced_vid_list,
        mask_output_dir=vid_mask_out_dir,
        seg_model=seg_model,
        seg_preprocess=seg_preprocess,
        person_idx=person_idx,
        seg_batch=2,
        mask_latent_size=(64, 64),
        device=device,
        # device="cpu",
        mask_reader_size=(mask_height, mask_width),
        skip_if_exists=skip_if_exists
    )
    assert len(
        vid_mask_list) == expected_num_samples, f"Number of video masks {len(vid_mask_list)} does not match number of data samples {expected_num_samples}"

    assert len(baseline_feat_sliced_audio) == len(jukebox_feat_sliced_audio) == len(vid_feats_list) == len(
        vid_mask_list) == len(alphapose_feat_list) == len(motionbert_feat_list), "Mismatch in number of extracted features"
    # return len(baseline_feat_sliced_audio), len(jukebox_feat_sliced_audio), len(vid_feats_list), len(vid_mask_list), len(alphapose_feat_list), len(motionbert_feat_list)
    return len(alphapose_feat_list), len(motionbert_feat_list)
