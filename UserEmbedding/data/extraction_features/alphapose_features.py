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
    vid_list: list,
    out_raw_json_dir: str=OUT_RAW_ROOT,
    out_json_dir: str=OUT_ROOT,
    cuda_device: str = None,
    conda_env: str = "alphapose",
) -> list:
    alphapose_out_list = []
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
    for vid_path in vid_list:
        vid_file_name = os.path.basename(vid_path)
        vid_name = os.path.splitext(vid_file_name)[0]
        raw_file_path = os.path.join(out_raw_json_dir, vid_name, "alphapose-results.json")
        final_file_path = os.path.join(out_json_dir, f"{vid_name}.json")
        print(f"process {vid_path}, vid name: {vid_file_name}, raw path: {raw_file_path}, final path: {final_file_path}")
        if os.path.exists(raw_file_path):
            print(f"Skip {vid_file_name}")
        else:
            cmd = [
                "conda", "run",
                "-n", conda_env,
                "python", ALPHA_POSE_FILE,
                "--cfg", CFG,
                "--checkpoint", CKPT,
                "--video", vid_path,
                "--outdir", os.path.join(out_raw_json_dir, vid_name),
                "--save_video",
            ]
            subprocess.run(
                cmd,
                cwd=AP_ROOT,
                env=env,
                check=True,
            )

        if os.path.exists(final_file_path):
            print(f"skip copy {final_file_path}")
        else:
            print(f"copy {final_file_path}")
            os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
            shutil.copyfile(raw_file_path, final_file_path)
        alphapose_out_list.append(final_file_path)

    return alphapose_out_list
            

if __name__ == "__main__":
    # Get CUDA device from command-line argument (default: "0")
    cuda_device = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Using CUDA device: {cuda_device}")
    run_alphapose(cuda_device)
