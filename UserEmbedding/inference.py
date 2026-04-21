#!/usr/bin/env python3
# infer_music_ref.py

import os
import glob
import random
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import cmp_to_key
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- project imports (adjust paths to match your repo) ---
from data.slice import slice_audio
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

from src.models import UserEmbeddingNet
from src.backbone import MotionBERTBackbone

from data.smpl_skeleton import SMPLSkeleton
from src.EDGE import DanceDecoder
from src.diffusion import GaussianDiffusion

# ---------------------------------------------------------

from data.run_alphapose import run_alphapose
from data.run_motionbert import run_motionbert
from data.slice import (
    slice_video,
    slice_motion_estimation,
    build_segmentation_model,
    slice_video_masks,
)

from videoprism import models as vp
from decord import VideoReader, cpu

VIDEOPRISM_MODEL_NAME = "videoprism_public_v1_base"

# same filename sorting logic as EDGE test
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


# ------------------------- reference video -------------------------


# TODO: replace this with video slicing/extraction code
def extract_ref_segments(
    root_dir: str,
    ref_video_dir: str,
    video_file: str,
    segment_frames: int,
    overlap_frames: float,
    cuda_device: torch.device,
    fps: int = 30,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Replace this with slice function.

    Must return list of segments, each segment is:
      (video_embedding, video_mask, pose_est)

    Shapes should match what UserEmbeddingNet expects, WITHOUT batch dim:
      video_embedding: (T, ...)
      video_mask:      (T,) or (T,1)
      pose_est:        (T, J, C) or similar

    T == segment_frames for every segment (recommended).
    All tensors should be on `device`.
    """

    # TODO: đợi cát bùi thêm overlap
    # TODO: slice motionbert result

    vr = VideoReader(f"{ref_video_dir}/{video_file}", ctx=cpu(0))
    original_fps = vr.get_avg_fps()
    step = int(original_fps // fps)

    video_slices = slice_video(
        video_path=f"{ref_video_dir}/{video_file}",
        length_frames=segment_frames,
        step=step,
        feature_output_dir=f"{root_dir}/video_features_sliced/{os.path.basename(video_file).split('.')[0]}",
        videoprism_model=VIDEOPRISM_MODEL_NAME,
        use_bfloat16=True,
    )

    motion_slices = slice_motion_estimation(
        pose_path=f"{root_dir}/pose_estimation/{os.path.basename(video_file).split('.')[0]}.json",
        length_frames=segment_frames,
        step=step,
        output_dir=f"{root_dir}/pose_estimation_sliced/{os.path.basename(video_file).split('.')[0]}",
    )

    seg_model, seg_preprocess, person_idx = build_segmentation_model(device=cuda_device)

    mask_slices = slice_video_masks(
        video_path=f"{ref_video_dir}/{video_file}",
        length_frames=segment_frames,
        step=step,
        mask_output_dir=f"{root_dir}/video_mask_sliced/{os.path.basename(video_file).split('.')[0]}",
        seg_model=seg_model,
        seg_preprocess=seg_preprocess,
        person_idx=person_idx,
        seg_batch=32,
        mask_latent_size=(64, 64),
        device=cuda_device,  # e.g. (720, 1280) or None for original
    )

    assert (
        video_slices == motion_slices == mask_slices
    ), f"Slicing mismatch: {len(video_slices)} vs {len(motion_slices)} vs {len(mask_slices)}"


@torch.no_grad()
def segment_embeddings_from_ref(
    user_embedding_net: UserEmbeddingNet,
    root_dir: str,
    video_file: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run UserEmbeddingNet per segment.
    Returns:
      seg_embs: (S, D)
      seg_w:    (S,) weights from mask coverage
    """
    seg_embs = []
    seg_w = []

    user_embedding_net.eval().to(device)

    reference_video_name = os.path.basename(video_file).split(".")[0]
    video_embedding_dir = f"{root_dir}/video_features_sliced/{reference_video_name}"
    video_mask_dir = f"{root_dir}/video_mask_sliced/{reference_video_name}"
    pose_est_dir = f"{root_dir}/pose_estimation_sliced/{reference_video_name}"

    video_embeddings = sorted(glob.glob(f"{video_embedding_dir}/*.npy"))
    video_masks = sorted(glob.glob(f"{video_mask_dir}/*.npy"))
    pose_ests = sorted(glob.glob(f"{pose_est_dir}/*.json"))

    for video_embedding, video_mask, pose_est in zip(
        video_embeddings, video_masks, pose_ests
    ):
        v = video_embedding.unsqueeze(0)  # (1, T, ...)
        m = video_mask.unsqueeze(0)  # (1, T) or (1, T, 1)
        p = pose_est.unsqueeze(0)  # (1, T, ...)

        embs, dancer_logits, pose_recon = user_embedding_net(v, m, p)
        embs = embs.squeeze(0).float()  # (D,)
        seg_embs.append(embs)

        mm2 = m.float()
        if mm2.dim() == 3:
            mm2 = mm2.squeeze(-1)
        seg_w.append(mm2.mean().clamp_min(1e-6).item())

    seg_embs = torch.stack(seg_embs, dim=0)  # (S, D)
    seg_w = torch.tensor(seg_w, device=device, dtype=torch.float32)  # (S,)
    return seg_embs, seg_w


def aggregate_segment_embeddings(
    seg_embs: torch.Tensor,  # (S, D)
    seg_w: torch.Tensor,  # (S,)
    method: str = "trimmed_cosine_consensus",
    trim_q: float = 0.2,
) -> torch.Tensor:
    """
    Aggregate (S,D)->(D,).
    Default: trimmed cosine-consensus (robust to bad/occluded segments).
    """
    S, D = seg_embs.shape
    if S == 1:
        return F.normalize(seg_embs[0], dim=0)

    e = F.normalize(seg_embs, dim=1)
    w = seg_w.clamp_min(1e-6)

    if method in ("mean", "weighted_mean"):
        mu = (e * w[:, None]).sum(dim=0) / w.sum()
        return F.normalize(mu, dim=0)

    if method == "trimmed_cosine_consensus":
        mu0 = (e * w[:, None]).sum(dim=0) / w.sum()
        mu0 = F.normalize(mu0, dim=0)

        cos = (e * mu0[None, :]).sum(dim=1)  # (S,)
        keep_k = max(1, int(round(S * (1.0 - trim_q))))
        keep_idx = torch.topk(cos, k=keep_k, largest=True).indices

        e2 = e[keep_idx]
        w2 = w[keep_idx]
        mu = (e2 * w2[:, None]).sum(dim=0) / w2.sum()
        return F.normalize(mu, dim=0)

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def build_single_dancer_embedding(
    root_dir: str,
    ref_video_dir: str,
    video_file: str,
    user_embedding_net: UserEmbeddingNet,
    device: torch.device,
    segment_frames: int,
    overlap_frames: float,
    agg_method: str,
    trim_q: float,
    fps: int = 30,
) -> torch.Tensor:
    extract_ref_segments(
        root_dir, ref_video_dir, video_file, segment_frames, overlap_frames, device, fps
    )
    seg_embs, seg_w = segment_embeddings_from_ref(
        user_embedding_net, root_dir, video_file, device
    )
    dancer_emb = aggregate_segment_embeddings(
        seg_embs, seg_w, method=agg_method, trim_q=trim_q
    )
    return dancer_emb  # (D,)


# ------------------------- checkpoint loading -------------------------


def load_matching_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """
    Loads only keys that exist AND match shape. Avoids classifier head size mismatch.
    """
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # (optional) print a short summary
    print(
        f"[Load] loaded {len(filtered)}/{len(state_dict)} tensors into {model.__class__.__name__}"
    )


def build_models(device: torch.device, feature_type: str):
    use_baseline_feats = feature_type == "baseline"
    feature_dim = 35 if use_baseline_feats else 4800

    # same dims as your train.py
    pos_dim = 3
    rot_dim = 24 * 6
    repr_dim = pos_dim + rot_dim + 4

    motionbert = MotionBERTBackbone()

    # num_dancer_class is irrelevant for inference output; we will load only matching weights
    user_embedding_net = UserEmbeddingNet(motionbert, num_dancer_class=1).to(device)

    edge = DanceDecoder(
        nfeats=repr_dim,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=feature_dim,
        activation=F.gelu,
    ).to(device)

    smpl = SMPLSkeleton(device)
    diffusion = GaussianDiffusion(
        edge,
        repr_dim,
        smpl,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=2,
    ).to(device)

    diffusion.model = edge
    return user_embedding_net, edge, diffusion, repr_dim


@torch.no_grad()
def load_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "emb_model" not in ckpt or "edge_model" not in ckpt:
        raise ValueError(f"Checkpoint missing keys. Found: {list(ckpt.keys())}")
    normalizer = ckpt.get("normalizer", None)
    step = ckpt.get("step", None)
    print(f"[CKPT] loaded {ckpt_path} (step={step})")
    return ckpt["emb_model"], ckpt["edge_model"], normalizer


# ------------------------- music processing -------------------------


def clamp_rand_start(n_items: int, chunk_size: int) -> int:
    if n_items <= chunk_size:
        return 0
    return random.randint(0, n_items - chunk_size)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--log_dir", type=str, default="logs/user_embedding")
    ap.add_argument(
        "--ckpt", type=str, required=True, help="ckp_*.pt saved by your train.py"
    )
    ap.add_argument(
        "--feature_type", type=str, default="jukebox", choices=["jukebox", "baseline"]
    )

    ap.add_argument("--root_dir", default="inference/", type=str, required=True)

    ap.add_argument("--render_dir", type=str, default="inference/result/render_result")
    ap.add_argument(
        "--motion_save_dir", type=str, default="inference/result/motion_result"
    )
    ap.add_argument("--render", action="store_true")
    ap.add_argument(
        "--sound", action="store_true", help="if your render_sample supports it"
    )

    ap.add_argument("--overlap_frames", type=float, default=0.5)
    ap.add_argument("--slice_frames", type=float, default=243)

    ap.add_argument("--cache_features", action="store_true")
    ap.add_argument("--use_cached_features", action="store_true")
    ap.add_argument("--feature_cache_dir", type=str, default="inference/backup_data")

    ap.add_argument("--video_fps", type=float, default=60)
    ap.add_argument(
        "--agg_method",
        type=str,
        default="trimmed_cosine_consensus",
        choices=["trimmed_cosine_consensus", "mean", "weighted_mean"],
    )
    ap.add_argument("--trim_q", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda")

    return ap.parse_args()


@torch.no_grad()
def main():
    opt = parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    Path(opt.render_dir).mkdir(parents=True, exist_ok=True)
    Path(opt.motion_save_dir).mkdir(parents=True, exist_ok=True)

    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract

    # build models like train.py
    user_embedding_net, edge, diffusion, repr_dim = build_models(
        device, opt.feature_type
    )

    # load checkpoint
    ckpt_path = os.path.join(opt.log_dir, opt.ckpt)
    emb_sd, edge_sd, normalizer = load_ckpt(ckpt_path, device)
    load_matching_state_dict(user_embedding_net, emb_sd)
    load_matching_state_dict(edge, edge_sd)
    diffusion.model = edge
    user_embedding_net.eval()
    diffusion.eval()

    # reference -> single embedding
    segment_frames = opt.slice_frames
    print(f"[Ref] segment_frames={segment_frames}")

    ref_video_dir = os.path.join(opt.root_dir, "video")
    wav_dir = os.path.join(opt.root_dir, "wavs")

    run_alphapose(
        device,
        vid_dir=ref_video_dir,
        out_raw_json_path=f"{opt.root_dir}/alphapose_raw_json",
        out_json_path=f"{opt.root_dir}/alphapose_json",
    )

    run_motionbert(
        device,
        video_dir=ref_video_dir,
        alphapose_dir=f"{opt.root_dir}/alphapose_json",
        motionbert_dir=f"{opt.root_dir}/motion_estimation",
    )

    # iterate songs
    tmp_dirs: List[TemporaryDirectory] = []
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    rederence_videos = glob.glob(os.path.join(ref_video_dir, "*.mp4"))

    assert len(wav_files) > 0, f"No .wav found in {wav_dir}"
    assert len(rederence_videos) > 0, f"No .mp4 found in {ref_video_dir}"

    assert len(rederence_videos) == len(
        wav_files
    ), f"Number of reference videos ({len(rederence_videos)}) must match number of wav files ({len(wav_files)})"

    for wav_file, video_reference in tqdm(
        zip(wav_files, rederence_videos), total=len(wav_files), desc="Processing songs"
    ):
        wav_name = os.path.splitext(os.path.basename(wav_file))[0]
        reference_name = os.path.splitext(os.path.basename(video_reference))[0]

        assert (
            wav_name == reference_name
        ), f"Mismatched wav and reference video: {wav_name} vs {reference_name}"

        print(f"\n=== {wav_name} ===")

        dancer_emb = build_single_dancer_embedding(
            opt.root_dir,
            ref_video_dir,
            video_reference,
            user_embedding_net,
            device=device,
            segment_frames=segment_frames,
            overlap_frames=opt.overlap_frames,
            agg_method=opt.agg_method,
            trim_q=opt.trim_q,
            fps=opt.video_fps,
        )  # (D,)
        print(f"[Ref] dancer_emb: {tuple(dancer_emb.shape)}")

        # choose working dir
        if opt.use_cached_features or opt.cache_features:
            song_dir = os.path.join(opt.feature_cache_dir, wav_name)
            Path(song_dir).mkdir(parents=True, exist_ok=True)
        else:
            td = TemporaryDirectory()
            tmp_dirs.append(td)
            song_dir = td.name

        # slice audio if needed
        if not opt.use_cached_features:
            print(f"[Music] slicing {wav_file} -> {song_dir}")
            # TODO: Add overlap
            slice_audio(
                os.path.join(wav_dir, wav_file),
                segment_frames,
                f"{opt.root_dir}/wavs_sliced/{wav_name}",
                opt.video_fps,
            )

        slice_wavs = sorted(
            glob.glob(f"{opt.root_dir}/wavs_sliced/{wav_name}"), key=stringintkey
        )

        if len(slice_wavs) == 0:
            print("[Music] WARNING: no slices produced, skipping.")
            continue

        # build cond stack
        cond_list = []
        names = []
        print(f"[Music] extracting features for {len(slice_wavs)} slices)")

        for f in tqdm(slice_wavs, desc="Extracting features"):
            npy_path = os.path.splitext(f)[0] + ".npy"

            if opt.use_cached_features and os.path.isfile(npy_path):
                reps = np.load(npy_path)
            else:
                reps, _ = feature_func(f)
                if opt.cache_features:
                    np.save(npy_path, reps)

            cond_list.append(torch.from_numpy(reps).float())
            names.append(os.path.basename(f))

        # cond: (B, T, F)
        cond = torch.stack(cond_list, dim=0).to(device)

        # embeddings: (B, D) — same dancer for all slices
        B, T, _ = cond.shape
        embs = dancer_emb.unsqueeze(0).expand(B, -1).contiguous().to(device)

        shape = (B, T, repr_dim)

        print(f"[Gen] cond={tuple(cond.shape)} embs={tuple(embs.shape)} shape={shape}")

        # call EXACTLY like your train/eval code
        diffusion.render_sample(
            shape,
            cond,
            embs,
            normalizer,
            label="test",
            render_dir=opt.render_dir,
            fk_out=opt.motion_save_dir,
            name=names,
            mode="long",
            sound=opt.sound,
            render=opt.render,
        )

    # cleanup
    for td in tmp_dirs:
        td.cleanup()

    print("\n[Done]")


if __name__ == "__main__":
    main()
