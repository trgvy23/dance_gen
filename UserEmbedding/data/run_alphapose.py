import subprocess
import os
import shutil
import sys
# from typing import final

# VID_DIR = "/raid/ltnghia02/vyttt/EDGE_modify/data/aist_plusplus_final/video"
# OUT_RAW_ROOT = "/raid/ltnghia02/vyttt/EDGE_modify/data/aist_plusplus_final/pose_estimation_raw"
# OUT_ROOT = "/raid/ltnghia02/vyttt/EDGE_modify/data/aist_plusplus_final/pose_estimation"
VID_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video"
OUT_RAW_ROOT = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation_raw"
OUT_ROOT = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/pose_estimation"
AP_ROOT = "/raid/ltnghia02/vyttt/AlphaPose"
ALPHA_POSE_FILE = f"{AP_ROOT}/scripts/demo_inference.py"
CFG = f"{AP_ROOT}/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
CKPT = f"{AP_ROOT}/pretrained_models/halpe26_fast_res50_256x192.pth"

def run_alphapose(
    cuda_device,
    vid_dir=VID_DIR,
    out_raw_json_path=OUT_RAW_ROOT,
    out_json_path=OUT_ROOT
):
    VID_EXT = (".mp4", ".avi", ".mov", ".mkv")
    print(f"list {os.walk(vid_dir)} at {vid_dir}")
    for root, dirs, files in os.walk(vid_dir):
        print(f"process {root}, {files}")
        if not dirs and not files:
            print("   → This directory is empty.")
        for file in files:
            print(f"    process {file}")
            if not file.lower().endswith(VID_EXT):
                continue
            vid_path = os.path.join(root, file)
            vid_name = os.path.splitext(file)[0]
            raw_file_path = os.path.join(out_raw_json_path, vid_name, "alphapose-results.json")
            final_file_path = os.path.join(out_json_path, f"{vid_name}.json")
            if os.path.exists(raw_file_path):
                print(f"Skip {vid_name}")
            #if not os.path.exists(raw_file_path):
            else:
                print(f"Processing {vid_name}...")
                cmd = [
                    "python", ALPHA_POSE_FILE,
                    "--cfg", CFG,
                    "--checkpoint", CKPT,
                    "--video", vid_path,
                    "--outdir", f"{out_raw_json_path}/{vid_name}",
                    "--save_video"
                ]
                if cuda_device:
                    print(f"use cuda device: {cuda_device}")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = cuda_device
                    subprocess.run(cmd, env=env, cwd=AP_ROOT)
                else:
                    print(f"not use cuda device")
                    subprocess.run(cmd, cwd=AP_ROOT)
            if os.path.exists(final_file_path):
                print(f"skip copy {final_file_path}")
            #if not os.path.exists(final_file_path):
            else:
                print(f"copy {final_file_path}")
                os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
                shutil.copyfile(raw_file_path, final_file_path)
            

if __name__ == "__main__":
    # Get CUDA device from command-line argument (default: "0")
    cuda_device = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Using CUDA device: {cuda_device}")
    run_alphapose(cuda_device)
