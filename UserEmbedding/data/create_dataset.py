import argparse
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import subprocess

from slice import slice_dataset
import generate_pose_data


def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

#TODO: Uncomment phần này khi đã chia lại train / test split
# filter_list = set(fileToList("splits/ignore_list.txt"))
# train_list = set(fileToList("splits/crossmodal_train.txt"))
# test_list = set(fileToList("splits/crossmodal_test.txt"))

def load_and_normalize(path):
    return set(x.replace("_cAll_", "_c01_") for x in fileToList(path))

filter_list = load_and_normalize("splits/ignore_list.txt")
train_list  = load_and_normalize("splits/crossmodal_train.txt")
test_list   = load_and_normalize("splits/crossmodal_test.txt")


def create_dataset(opt):
    # split the data according to the splits files

    # run Alpha Pose to get 2D pose estimation
    generate_pose_data.generate_pose_estimation()

    #TODO: Chia lại train / test split. Hiện tại đang dùng split của AIST++ để code chạy được
    print("Creating train / test split")
    split_data(opt.dataset_folder)

    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_dataset(f"train/video", f"train/pose_estimation")
    print("Slicing test data")
    slice_dataset(f"test/video", f"test/pose_estimation")


def split_data(dataset_path):
    # train - test split
    skipping_files = 0
    
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{split_name}/video").mkdir(parents=True, exist_ok=True)
        Path(f"{split_name}/pose_estimation").mkdir(parents=True, exist_ok=True)

        for sequence in tqdm(split_list, desc=f"Processing {split_name} data"):
            if sequence in filter_list:
                continue

            video = f"{dataset_path}/video/{sequence}.mp4"
            pose_estimation = f"{dataset_path}/pose_estimation/{sequence}.json"

            #NOTE: Hiện tại pose estimation chưa đủ, nên tạm comment phần assert lại.
            #TODO: Uncomment lại khi đã có đủ pose estimation
            # assert os.path.isfile(video), f"Missing motion: {video}"
            # assert os.path.isfile(pose_estimation),    f"Missing wav: {pose_estimation}"

            if not os.path.isfile(video) or not os.path.isfile(pose_estimation):
                skipping_files += 1
                continue

            # copy
            shutil.copyfile(video, f"{split_name}/video/{sequence}.mp4")
            shutil.copyfile(
                pose_estimation, f"{split_name}/pose_estimation/{sequence}.json"
            )
            
    print(f"Skipped {skipping_files} files due to missing data.")


def parse_opt():
    # TODO: set width and height according to dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    # 680 480
    parser.add_argument(
        "--width", type=int, default=None, help="width to resize videos to"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="height to resize videos to"
    )
    parser.add_argument(
        "--length_frames",
        type=int,
        default=243,
        help="number of frames per slice",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frame rate to sample videos",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
