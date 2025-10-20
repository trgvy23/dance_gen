import argparse
import os
from pathlib import Path

import glob
import os
import pickle
import shutil
from pathlib import Path


def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out


filter_list = set(fileToList("splits/ignore_list.txt"))
train_list = set(fileToList("splits/crossmodal_train.txt"))
test_list = set(fileToList("splits/crossmodal_test.txt"))


def create_dataset(opt):
    # split the data according to the splits files
    print("Creating train / test split")
    split_data(opt.dataset_folder)

    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_data(f"train/video", f"train/post_estimation")
    print("Slicing test data")
    slice_data(f"test/video", f"test/post_estimation")


def slice_data(video_path, pose_estimation_path):
    pass


def split_data(dataset_path):
    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{split_name}/video").mkdir(parents=True, exist_ok=True)
        Path(f"{split_name}/post_estimation").mkdir(parents=True, exist_ok=True)

        for sequence in split_list:
            if sequence in filter_list:
                continue

            video = f"{dataset_path}/video/{sequence}.mp4"
            pose_estimation = f"{dataset_path}/post_estimation/{sequence}.json"

            # assert os.path.isfile(video), f"Missing motion: {video}"
            # assert os.path.isfile(pose_estimation),    f"Missing wav: {pose_estimation}"

            if not os.path.isfile(video) or not os.path.isfile(pose_estimation):
                continue

            # copy
            shutil.copyfile(video, f"{split_name}/video/{sequence}.mp4")
            shutil.copyfile(
                pose_estimation, f"{split_name}/post_estimation/{sequence}.json"
            )


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
