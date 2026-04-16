import argparse
from ctypes.util import test
import os
from pathlib import Path
import shutil
from unittest import skip
from tqdm import tqdm
import subprocess

from extracting_features.extracting_features import extract_data_feats
from io_files import copy_unique_music_by_id, load_data_ids, save_data_ids, ensure_dir, copy_dataset_with_shuffle
from shuffle import shuffle, rename_with_mapping_os
from split import split_from_list_data, add_test_from_train
from slice import slice_dataset


def split_data(
    train_data: list,
    test_data: list,
    ignore_data: list,
    train_data_file_path: str,
    test_data_file_path: str,
    ignore_data_file_path: str,
    org_data_folder: str,
    train_data_folder: str,
    test_data_folder: str,
):
    train_data, test_data, ignore_data = split_from_list_data(
        train_data, test_data, ignore_data)
    save_data_ids(train_data, train_data_file_path)
    save_data_ids(test_data, test_data_file_path)
    save_data_ids(ignore_data, ignore_data_file_path)

    assert os.path.exists(org_data_folder)
    org_vid_dir = os.path.join(org_data_folder, "video")
    assert os.path.exists(org_vid_dir)
    org_music_dir = os.path.join(org_data_folder, "wavs")
    assert os.path.exists(org_music_dir)
    org_motion_dir = os.path.join(org_data_folder, "motions")
    assert os.path.exists(org_motion_dir)

    train_vid_dir = os.path.join(train_data_folder, "video")
    ensure_dir(train_vid_dir, create_new=True)
    train_music_dir = os.path.join(train_data_folder, "wavs")
    ensure_dir(train_music_dir, create_new=True)
    train_motion_dir = os.path.join(train_data_folder, "motions")
    ensure_dir(train_motion_dir, create_new=True)

    test_vid_dir = os.path.join(test_data_folder, "video")
    ensure_dir(test_vid_dir, create_new=True)
    test_music_dir = os.path.join(test_data_folder, "wavs")
    ensure_dir(test_music_dir, create_new=True)
    test_motion_dir = os.path.join(test_data_folder, "motions")
    ensure_dir(test_motion_dir, create_new=True)

    train_ids = set(train_data)
    test_ids = set(test_data)

    # copy videos from dataset
    for file in os.scandir(org_vid_dir):
        if file.is_file():
            file_name = os.path.splitext(os.path.basename(file))[0]
            if file_name in train_ids:
                shutil.copy(file.path, os.path.join(
                    train_vid_dir, file_name + ".mp4"))
            elif file_name in test_ids:
                shutil.copy(file.path, os.path.join(
                    test_vid_dir, file_name + ".mp4"))

    # copy music from dataset
    for file in os.scandir(org_music_dir):
        if file.is_file():
            file_name = os.path.splitext(os.path.basename(file))[0]
            if file_name in train_ids:
                shutil.copy(file.path, os.path.join(
                    train_music_dir, file_name + ".wav"))
            elif file_name in test_ids:
                shutil.copy(file.path, os.path.join(
                    test_music_dir, file_name + ".wav"))

    # copy motion from dataset
    for file in os.scandir(org_motion_dir):
        if file.is_file():
            file_name = os.path.splitext(os.path.basename(file))[0]
            if file_name in train_ids:
                shutil.copy(file.path, os.path.join(
                    train_motion_dir, file_name + ".pkl"))
            elif file_name in test_ids:
                shutil.copy(file.path, os.path.join(
                    test_motion_dir, file_name + ".pkl"))

    return train_data, test_data, ignore_data


def create_dataset(opt):
    train_data_file_path = opt.train_data_file
    test_data_file_path = opt.test_data_file
    org_data_folder = opt.dataset_folder
    train_data_folder = opt.train_folder
    test_data_folder = opt.test_folder
    train_data_list = load_data_ids(train_data_file_path)
    test_data_list = load_data_ids(test_data_file_path)
    ignore_data_list = load_data_ids(
        opt.ignore_data_file) if opt.ignore_data_file != "" else []

    if opt.do_shuffle:
        print("Splitting dataset into train and test sets")
        train_data_list, test_data_list, ignore_data_list = split_data(train_data_list, test_data_list, ignore_data_list, train_data_file_path,
                                                                       test_data_file_path, opt.ignore_data_file, org_data_folder, train_data_folder, test_data_folder)
        if len(train_data_list) < len(test_data_list):
            train_data_list, test_data_list = list(
                test_data_list), list(train_data_list)

    print("Slicing train set")
    sliced_train_data_list = slice_dataset(train_data_folder, train_data_list, length_frames=opt.length_frames,
                                        #    fps=opt.fps, overlap=opt.overlap, skip_if_exists=False)
                                           fps = opt.fps, overlap = opt.overlap, skip_if_exists = not opt.do_shuffle)

    print("Slicing test set")
    sliced_test_data_list = slice_dataset(test_data_folder, test_data_list, length_frames=opt.length_frames,
                                        #   fps=opt.fps, overlap=opt.overlap, skip_if_exists=False)
                                          fps = opt.fps, overlap = opt.overlap, skip_if_exists = not opt.do_shuffle)

    if opt.do_shuffle:
        print("Shuffling music IDs")
        sliced_train_data_list, sliced_test_data_list, shuffle_map = shuffle(
            sliced_train_data_list, sliced_test_data_list, opt.shuffling_map_file)
        print("Renaming files according to shuffle map")
        rename_with_mapping_os(os.path.join(
            train_data_folder, "wavs_sliced"), shuffle_map,)
        rename_with_mapping_os(os.path.join(
            test_data_folder, "wavs_sliced"), shuffle_map,)

    print("Extracting features for train set")
    extract_data_feats(train_data_folder, sliced_train_data_list, mask_width=opt.width, mask_height=opt.height,
                       skip_if_exists=not opt.do_shuffle, alphapose_env=opt.alphapose_env, motionbert_env=opt.motionbert_env)
    print("Extracting features for test set")
    extract_data_feats(test_data_folder, sliced_test_data_list, mask_width=opt.width, mask_height=opt.height,
                       skip_if_exists=not opt.do_shuffle, alphapose_env=opt.alphapose_env, motionbert_env=opt.motionbert_env)
    pass


def parse_opt():
    # TODO: set width and height according to dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        default="train",
        help="folder containing train data",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default="test",
        help="folder containing test data",
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="splits/tmp_train.txt",
        help="path to train list",
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default="splits/tmp_test.txt",
        help="path to test list",
    )
    parser.add_argument(
        "--ignore_data_file",
        type=str,
        default="",
        help="path to ignore list",
    )
    parser.add_argument(
        "--do_shuffle",
        action="store_true",
        default=False,
        help="do shuffle or not",
    )
    parser.add_argument(
        "--shuffling_map_file",
        type=str,
        default="",
        help="path to shuffling map",
    )
    # 680 480
    parser.add_argument(
        "--width", type=int, default=680, help="width to resize videos to"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="height to resize videos to"
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
    parser.add_argument(
        "--overlap",
        type=float,
        default=1.0,
        help="how much of the previous slice to overlap with the next slice",
    )
    parser.add_argument(
        "--alphapose_env",
        type=str,
        default="alphapose",
        help="conda environment to run AlphaPose inference in",
    )
    parser.add_argument(
        "--alphapose_dir",
        type=str,
        default="../AlphaPose",
        help="directory of AlphaPose code",
    )
    parser.add_argument(
        "--motionbert_env",
        type=str,
        default="motionbert",
        help="conda environment to run MotionBERT inference in",
    )
    parser.add_argument(
        "--motionbert_dir",
        type=str,
        default="../MotionBERT",
        help="directory of MotionBERT code",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
