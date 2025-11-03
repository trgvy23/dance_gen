import subprocess
import os
import shutil
from typing import final

VID_DIR = "/raid/ltnghia02/vyttt/EDGE_modify/data/aist_plusplus_final/video"
OUT_RAW_ROOT = "/raid/ltnghia02/vyttt/EDGE_modify/data/aist_plusplus_final/pose_estimation_raw"
OUT_ROOT = "/raid/ltnghia02/vyttt/EDGE_modify/data/aist_plusplus_final/pose_estimation"
AP_ROOT = "/raid/ltnghia02/vyttt/AlphaPose"
ALPHA_POSE_FILE = f"{AP_ROOT}/scripts/demo_inference.py"
CFG = f"{AP_ROOT}/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
CKPT = f"{AP_ROOT}/pretrained_models/halpe26_fast_res50_256x192.pth"

def generate_pose_estimation(
    vid_dir=VID_DIR,
    out_raw_json_path=OUT_RAW_ROOT,
    out_json_path=OUT_ROOT
):
    VID_EXT = (".mp4", ".avi", ".mov", ".mkv")
    for root, _, files in os.walk(vid_dir):
        for file in files:
            if not file.lower().endswith(VID_EXT):
                continue
            vid_path = os.path.join(root, file)
            vid_name = os.path.splitext(file)[0]
            raw_file_path = os.path.join(out_raw_json_path, vid_name, "alphapose-results.json")
            final_file_path = os.path.join(out_json_path, f"{vid_name}.json")
            if not os.path.exists(raw_file_path):
                print(f"Processing {vid_name}...")
                cmd = [
                    "python", ALPHA_POSE_FILE,
                    "--cfg", CFG,
                    "--checkpoint", CKPT,
                    "--video", vid_path,
                    "--outdir", f"{out_raw_json_path}/{vid_name}",
                    "--save_video"
                ]
                subprocess.run(cmd)
            if not os.path.exists(final_file_path):
                os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
                shutil.copyfile(raw_file_path, final_file_path)
            