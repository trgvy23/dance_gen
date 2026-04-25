import subprocess
import os
import shutil
import sys
import time
from tqdm import tqdm
import json
import re
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

SCORE_KEY = "score"
MAX_FRAMES = 243            # total frames
LAST_FRAME = MAX_FRAMES - 1  # 242
KEYPOINT_DIM = 78
ZERO_KEYPOINTS = [0] * KEYPOINT_DIM
# ==================

FRAME_RE = re.compile(r"(\d+)\.jpg")


def dedup_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return False, 0

    best_frames = {}   # image_id -> (frame, score)
    remove = 0

    for frame in data:
        if not isinstance(frame, dict):
            continue

        image_id = frame.get("image_id")
        if image_id is None:
            continue

        score = frame.get(SCORE_KEY, 0.0)
        if score is None:
            score = 0.0

        if image_id not in best_frames:
            best_frames[image_id] = (frame, score)
        else:
            _, prev_score = best_frames[image_id]
            if score > prev_score:
                best_frames[image_id] = (frame, score)
            remove += 1

    if remove == 0:
        return

    cleaned_data = []
    kept_ids = set()

    for frame in data:
        image_id = frame.get("image_id")
        if image_id in best_frames and image_id not in kept_ids:
            cleaned_data.append(best_frames[image_id][0])
            kept_ids.add(image_id)

    with open(json_path, "w") as f:
        json.dump(cleaned_data, f)


def extract_frame_id(image_id: str) -> int:
    m = FRAME_RE.search(image_id)
    if not m:
        raise ValueError(f"Cannot parse frame id from image_id: {image_id}")
    return int(m.group(1))


def make_image_id(frame_id: int) -> str:
    return f"{frame_id}.jpg"


def interpolate_keypoints(start_frame, end_frame):
    start_id = extract_frame_id(start_frame["image_id"])
    end_id = extract_frame_id(end_frame["image_id"])
    n_missing = end_id - start_id - 1
    if n_missing <= 0:
        return []
    start_kp = start_frame["keypoints"]
    end_kp = end_frame["keypoints"]

    interpolated = []
    for i in range(1, n_missing + 1):
        # tqdm.write(f"  Interpolating frame {make_image_id(start_id + i)}")
        ratio = i / (n_missing + 1)
        kp = [
            start_kp[j] * (1 - ratio) + end_kp[j] * ratio
            for j in range(len(start_kp))
        ]
        interpolated.append({
            "image_id": make_image_id(start_id + i),
            "keypoints": kp,
        })

    return interpolated


def fix_alphapose_frame_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    if not data:
        return False

    frame_map = {}
    for item in data:
        fid = extract_frame_id(item["image_id"])
        frame_map[fid] = item

    existing_frames = sorted(frame_map.keys())
    min_existing = existing_frames[0]
    max_existing = existing_frames[-1]
    tqdm.write(
        f"Processing {path}: existing frames {min_existing}..{max_existing}")

    new_data = []
    modified = False

    for fid in range(0, LAST_FRAME + 1):
        if fid in [extract_frame_id(i["image_id"]) for i in new_data]:
            continue
        if fid in frame_map:
            item = frame_map[fid]
            new_data.append(item)
            continue

        modified = True
        image_id = make_image_id(fid)

        # 1. missing at beginning
        if fid < min_existing:
            # tqdm.write(f"Filling missing starting frame {image_id} with zeros")
            new_item = {
                "image_id": image_id,
                "keypoints": ZERO_KEYPOINTS,
            }
            new_data.append(new_item)

        # 2. missing in the middle
        elif fid <= max_existing:
            next_existing_fid = fid + 1
            while next_existing_fid not in frame_map and next_existing_fid <= max_existing:
                next_existing_fid += 1

            if next_existing_fid <= max_existing:
                prev_item = frame_map[fid - 1]
                next_item = frame_map[next_existing_fid]
                # tqdm.write(
                #     f"Interpolating missing frame {image_id} between {prev_item['image_id']} and {next_item['image_id']}")
                interpolated = interpolate_keypoints(prev_item, next_item)
                new_data.extend(interpolated)
                new_data.append(next_item)
                new_data.sort(key=lambda x: extract_frame_id(x["image_id"]))

        # 3. missing at the end
        else:
            # tqdm.write(f"Filling missing ending frame {image_id} with zeros")
            new_item = {
                "image_id": image_id,
                "keypoints": ZERO_KEYPOINTS,
            }

            new_data.append(new_item)

    if modified:
        with open(path, "w") as f:
            json.dump(new_data, f)


def check_result_file(file_path: str):
    assert os.path.exists(file_path), f"Result file {file_path} does not exist"
    dedup_file(file_path)
    fix_alphapose_frame_json(file_path)


def run_alphapose(
    vid_list: list,
    out_raw_json_dir: str = OUT_RAW_ROOT,
    out_json_dir: str = OUT_ROOT,
    conda_env: str = "alphapose",
    alphapose_dir: str = AP_ROOT,
    skip_if_exists: bool = False,
) -> list:
    alphapose_out_list = []
    env = os.environ.copy()

    pbar = tqdm(vid_list, desc="AlphaPose inference")
    alphapose_script_path = os.path.join(
        alphapose_dir, "scripts/demo_inference.py")
    alphapose_cfg_path = os.path.join(
        alphapose_dir, "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml")
    alphapose_ckpt_path = os.path.join(
        alphapose_dir, "pretrained_models/halpe26_fast_res50_256x192.pth")
    for vid_path in pbar:
        vid_file_name = os.path.basename(vid_path)
        assert os.path.isfile(
            vid_path), f"Video file {vid_path} does not exist"
        vid_name = os.path.splitext(vid_file_name)[0]
        raw_file_path = os.path.join(
            out_raw_json_dir, vid_name, "alphapose-results.json")
        final_file_path = os.path.join(out_json_dir, f"{vid_name}.json")
        # print(f"process {vid_path}, vid name: {vid_file_name}, raw path: {raw_file_path}, final path: {final_file_path}")
        pbar.set_postfix({"file": vid_file_name})
        if skip_if_exists and os.path.exists(raw_file_path):
            tqdm.write(f"Skip {vid_file_name}")
        else:
            cmd = [
                "conda", "run",
                "-n", conda_env,
                "python", alphapose_script_path,
                "--cfg", alphapose_cfg_path,
                "--checkpoint", alphapose_ckpt_path,
                "--video", vid_path,
                "--outdir", os.path.join(out_raw_json_dir, vid_name),
                "--save_video",
            ]
            subprocess.run(
                cmd,
                cwd=alphapose_dir,
                env=env,
                check=True,
            )

        check_result_file(raw_file_path)
        if os.path.exists(final_file_path):
            # print(f"skip copy {final_file_path}")
            pass
        else:
            # print(f"copy {final_file_path}")
            os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
            shutil.copyfile(raw_file_path, final_file_path)
        alphapose_out_list.append(final_file_path)
        time.sleep(2)

    return alphapose_out_list


if __name__ == "__main__":
    # Get CUDA device from command-line argument (default: "0")
    cuda_device = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Using CUDA device: {cuda_device}")
    run_alphapose(cuda_device)
