import glob
import os
import pickle
import json
import re
from pathlib import Path
import copy
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from decord import VideoReader
from tqdm import tqdm


def halpe2h36m(x):
    """
        Input: x (T x V x C)
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    """
    T, V, C = x.shape
    y = np.zeros([T, 17, C])
    y[:, 0, :] = x[:, 19, :]
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 7, :] = (x[:, 18, :] + x[:, 19, :]) * 0.5
    y[:, 8, :] = x[:, 18, :]
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = x[:, 17, :]
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    return y


def crop_scale(motion, scale_range=[1, 1]):
    """
    Motion: [(M), T, 17, 3].
    Normalize to [-1, 1]
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


class DanceDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool = True,
        force_reload: bool = False,
        cache_data: bool = False,
    ):
        self.data_path = data_path
        # self.raw_fps = 60
        # self.data_fps = 30
        # assert self.data_fps <= self.raw_fps
        # self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"

        self.genre2id, self.dancer2id = {}, {}

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_aistpp()  # Call this last
            if cache_data:
                with open(os.path.join(backup_path, pickle_name), "wb") as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        logging.info(
            f"Loaded {self.name} Dataset With Dimensions: \n\tVideo embeddings: {data['video_embeddings'].shape}, \n\tPose Estimations: {data['pose_estimations'].shape}"
        )

        self.data = {
            "video_embeddings": data["video_embeddings"],
            "pose_estimations": data["pose_estimations"],
            "gerne_labels": data["gerne_labels"],
            "dancer_labels": data["dancer_labels"],
        }

        assert len(data["video_embeddings"]) == len(data["pose_estimations"])
        self.length = len(data["video_embeddings"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.data["video_embeddings"][idx],
            self.data["pose_estimations"][idx],
            self.data["gerne_labels"][idx],
            self.data["dancer_labels"][idx],
        )
        
    def get_dancer_num(self):
        labels = self.data["dancer_labels"]
        if isinstance(labels, np.ndarray):
            return int(labels.max()) + 1
        return int(labels.max().item()) + 1
    
    def get_gerne_num(self):
        labels = self.data["gerne_labels"]
        if isinstance(labels, np.ndarray):
            return int(labels.max()) + 1
        return int(labels.max().item()) + 1
    
    def read_video(self, video_embedding_path: str):
        return np.load(video_embedding_path, allow_pickle=True)  # (T, H, W, 3)

    def read_pose_estimation(self, json_path, vid_size=None):
        with open(json_path, "r") as read_file:
            results = json.load(read_file)
        kpts_all = []
        for item in results:
            kpts = np.array(item["keypoints"]).reshape([-1, 3])
            kpts_all.append(kpts)
        kpts_all = np.array(kpts_all)
        kpts_all = halpe2h36m(kpts_all)
        if vid_size:
            w, h = vid_size
            scale = min(w, h) / 2.0
            kpts_all[:, :, :2] = kpts_all[:, :, :2] - np.array([w, h]) / 2.0
            kpts_all[:, :, :2] = kpts_all[:, :, :2] / scale
            pose_estimation = kpts_all
        else:
            pose_estimation = crop_scale(kpts_all)
        return pose_estimation.astype(np.float32)  # (T, 17, 3)

    def parse_aist_labels(self, filename):
        # name: "gBR_sBM_cAll_d04_mBR0_ch02"
        m = re.match(r"^(g[A-Z]{2})_.*_m([A-Z]{2}\d+)_", filename)
        assert m, f"Bad AIST++ name: {filename}"
        genre_code = m.group(1)  # e.g., gBR
        dancer_code = m.group(2)  # e.g., BR0
        return genre_code, dancer_code

    def read_label(self, filename):
        genre_code, dancer_code = self.parse_aist_labels(filename)
        if genre_code not in self.genre2id:
            self.genre2id[genre_code] = len(self.genre2id)
        if dancer_code not in self.dancer2id:
            self.dancer2id[dancer_code] = len(self.dancer2id)
        genre_id = self.genre2id[genre_code]
        dancer_id = self.dancer2id[dancer_code]
        return genre_id, dancer_id

    def load_aistpp(self):
        # open data path
        split_root = os.path.join(self.data_path, "train" if self.train else "test")

        # Structure:
        # data
        #   |- train
        #   |    |- videos
        #   |    |- pose_estimation

        video_embedding_path = os.path.join(split_root, "video_embedding_sliced")
        pose_estimation_path = os.path.join(split_root, "pose_estimation_sliced")

        # sort motions and sounds
        video_embeddings = sorted(glob.glob(os.path.join(video_embedding_path, "*.npy")))
        pose_estimations = sorted(
            glob.glob(os.path.join(pose_estimation_path, "*.json"))
        )

        assert len(video_embeddings) == len(
            pose_estimations
        ), f"Count mismatch: video_embeddings={len(video_embeddings)} pose_estimations={len(pose_estimations)}"

        all_video_embeddings, all_pose_estimations, all_gerne_labels, all_dancer_labels = (
            [],
            [],
            [],
            [],
        )

        for video_embedding_filename, pose_est_filename in tqdm(
            zip(video_embeddings, pose_estimations), total=len(video_embeddings), desc="Loading data"
        ):
            v_name = os.path.splitext(os.path.basename(video_embedding_filename))[0]
            p_name = os.path.splitext(os.path.basename(pose_est_filename))[0]
            assert (
                v_name == p_name
            ), f"Name mismatch: {video_embedding_filename} vs {pose_est_filename}"

            video_embedding = self.read_video(video_embedding_filename)
            all_video_embeddings.append(video_embedding)

            # video_height = video_embedding.shape[1]
            # video_width = video_embedding.shape[2]

            pose_est = self.read_pose_estimation(
                pose_est_filename, vid_size=None
            )
            all_pose_estimations.append(pose_est)

            genre_id, dancer_id = self.read_label(v_name)
            all_gerne_labels.append(genre_id)
            all_dancer_labels.append(dancer_id)

        all_video_embeddings = np.array(all_video_embeddings)  # N x T x H x W x 3
        all_pose_estimations = np.array(all_pose_estimations)  # N x T x 17 x 3
        all_gerne_labels = np.array(all_gerne_labels)  # N
        all_dancer_labels = np.array(all_dancer_labels)  # N

        data = {
            "video_embeddings": all_video_embeddings,
            "pose_estimations": all_pose_estimations,
            "gerne_labels": all_gerne_labels,
            "dancer_labels": all_dancer_labels,
        }
        return data
