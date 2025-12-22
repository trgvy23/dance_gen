import os
import argparse
import subprocess

# ==== DEFAULT PATHS ====
VIDEO_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video"
ALPHAPOSE_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/alpha_pose"
MOTIONBERT_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation"
MB_ROOT = "/raid/ltnghia02/vyttt/MotionBERT"
CONFIG = "configs/pose3d/MB_ft_h36m_global_lite.yaml"
CHECKPOINT = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"


def run_motionbert(cuda_device, video_dir=VIDEO_DIR, alphapose_dir=ALPHAPOSE_DIR, motionbert_dir=MOTIONBERT_DIR):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device

    os.makedirs(motionbert_dir, exist_ok=True)

    for file in os.listdir(alphapose_dir):
        if not file.endswith(".json"):
            continue

        base_name = os.path.splitext(file)[0]
        video_path = os.path.join(video_dir, f"{base_name}.mp4")
        json_path = os.path.join(alphapose_dir, file)

        if not os.path.exists(video_path):
            print(f"Video not found for {base_name}, skipping...")
            continue

        print(f"Running MotionBERT on: {video_path}")
        cmd = [
            "python", "infer_wild_custom.py",
            "--config", CONFIG,
            "--evaluate", CHECKPOINT,
            "--vid_path", video_path,
            "--json_path", json_path,
            "--out_path", motionbert_dir,
            "--clip_len", "243"
        ]

        subprocess.run(cmd, env=env, cwd=MB_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run MotionBERT inference on videos + AlphaPose JSONs.")
    parser.add_argument("--cuda_device", default="0", help="CUDA device ID to use (default: 0)")
    parser.add_argument("--video_dir", default=VIDEO_DIR, help="Directory containing videos")
    parser.add_argument("--alphapose_dir", default=ALPHAPOSE_DIR, help="Directory containing AlphaPose results (.json)")
    parser.add_argument("--motionbert_dir", default=MOTIONBERT_DIR, help="Directory to save MotionBERT outputs")

    args = parser.parse_args()

    run_motionbert(
        args.cuda_device,
        video_dir=args.video_dir,
        alphapose_dir=args.alphapose_dir,
        motionbert_dir=args.motionbert_dir
    )


if __name__ == "__main__":
    main()
