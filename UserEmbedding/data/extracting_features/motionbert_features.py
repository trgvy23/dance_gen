import os
import argparse
import subprocess
import time
from tqdm import tqdm

# ==== DEFAULT PATHS ====
ROOT_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/"
VIDEO_DIR = f"{ROOT_DIR}/video_sliced"
ALPHAPOSE_DIR = f"{ROOT_DIR}/alphapose_sliced"
MOTIONBERT_DIR = f"{ROOT_DIR}/pose_estimation"
MB_ROOT = "/raid/ltnghia02/vyttt/MotionBERT"
CONFIG = "configs/pose3d/MB_ft_h36m_global_lite.yaml"
CHECKPOINT = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"


def run_motionbert(
    sliced_video_list: list,
    sliced_alphapose_list: list,
    sliced_motionbert_dir: str,
    conda_env: str = "motionbert",
    motionbert_dir: str = MB_ROOT,
    skip_if_exists: bool = False,
) -> list:
    """
    Run MotionBERT inference on sliced videos + AlphaPose jsons.

    Args:
        sliced_video_list: list of sliced .mp4 paths
        sliced_alphapose_list: list of corresponding alphapose .json paths
        sliced_motionbert_dir: directory to save MotionBERT outputs
        conda_env: conda environment name for MotionBERT

    Returns:
        list of output file paths
    """
    assert len(sliced_video_list) == len(sliced_alphapose_list), \
        "Mismatch between video list and alphapose list"

    os.makedirs(sliced_motionbert_dir, exist_ok=True)

    output_list = []
    env = os.environ.copy()

    pbar = tqdm(
        zip(sliced_video_list, sliced_alphapose_list),
        total=len(sliced_video_list),
        desc="MotionBERT inference",
    )
    motionbert_cfg_path = os.path.join(motionbert_dir, CONFIG)
    motionbert_ckpt_path = os.path.join(motionbert_dir, CHECKPOINT)
    for video_path, json_path in pbar:
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # MotionBERT usually outputs per-video files
        out_path = os.path.join(sliced_motionbert_dir, f"{base_name}.pkl")

        # print(f"\nProcessing MotionBERT: {base_name}")
        # print(f"  video: {video_path}")
        # print(f"  pose : {json_path}")
        # print(f"  out  : {out_path}")

        pbar.set_postfix({"file": base_name})
        if skip_if_exists and os.path.exists(out_path):
            tqdm.write(f"Skip {base_name} (already exists)")
            output_list.append(out_path)
            continue

        if not os.path.exists(video_path):
            tqdm.write(f"Video not found, skipping {base_name}")
            continue

        if not os.path.exists(json_path):
            tqdm.write(f"AlphaPose json not found, skipping {base_name}")
            continue

        cmd = [
            "conda", "run", "-n", conda_env,
            "python", "infer_wild_custom.py",
            "--config", motionbert_cfg_path,
            "--evaluate", motionbert_ckpt_path,
            "--vid_path", video_path,
            "--json_path", json_path,
            "--out_path", sliced_motionbert_dir,
            "--clip_len", "243",
        ]

        subprocess.run(
            cmd,
            cwd=motionbert_dir,
            env=env,
            check=True,
        )

        if not os.path.exists(out_path):
            raise RuntimeError(
                f"MotionBERT failed, output not found:\n{out_path}"
            )

        output_list.append(out_path)
        time.sleep(2)

    return output_list


def main():
    parser = argparse.ArgumentParser(
        description="Run MotionBERT inference on videos + AlphaPose JSONs.")
    parser.add_argument("--cuda_device", default="0",
                        help="CUDA device ID to use (default: 0)")
    parser.add_argument("--video_dir", default=VIDEO_DIR,
                        help="Directory containing videos")
    parser.add_argument("--alphapose_dir", default=ALPHAPOSE_DIR,
                        help="Directory containing AlphaPose results (.json)")
    parser.add_argument("--motionbert_dir", default=MOTIONBERT_DIR,
                        help="Directory to save MotionBERT outputs")

    args = parser.parse_args()

    run_motionbert(
        args.cuda_device,
        video_dir=args.video_dir,
        alphapose_dir=args.alphapose_dir,
        motionbert_dir=args.motionbert_dir
    )


if __name__ == "__main__":
    main()
