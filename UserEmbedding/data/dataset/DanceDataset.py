import glob
import os
import pickle
import json
from pathlib import Path
import copy
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any
from decord import VideoReader
from tqdm import tqdm
from data.smpl_skeleton import SMPLSkeleton
from pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from data.preprocess import Normalizer, vectorize_many
from data.quaternion import ax_to_6v
from .dataset_utils import (
    halpe2h36m,
    crop_scale,
    parse_aist_labels_from_name,
    build_label_mappings,
)


class DanceDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool = True,
        force_reload: bool = False,
        cache_data: bool = False,
        genre2id=None,
        dancer2id=None,
        feature_type: str = "jukebox",
        normalizer: Any = None,
    ):
        self.data_path = data_path
        self.raw_fps = 60
        self.data_fps = 30
        self.data_stride = self.raw_fps // self.data_fps
        # self.raw_fps = 60
        # self.data_fps = 30
        # assert self.data_fps <= self.raw_fps
        # self.data_stride = self.raw_fps // self.data_fps
        self.feature_type = feature_type

        self.train = train
        self.name = "Train" if self.train else "Test"

        # If provided, we treat them as fixed global mappings
        self.genre2id = {} if genre2id is None else genre2id
        self.dancer2id = {} if dancer2id is None else dancer2id
        self.fixed_label_maps = (genre2id is not None) and (
            dancer2id is not None)

        self.normalizer = normalizer

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        if not train:
            pickle.dump(
                normalizer, open(os.path.join(
                    backup_path, "normalizer.pkl"), "wb")
            )

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
            f"Loaded {self.name} Dataset With Dimensions: \n\tVideo embeddings: {data['video_embeddings'].shape}, \n\tPose Estimations: {data['pose_estimations'].shape}, \n\tPose Masks: {data['video_masks'].shape}, \n\tGenre Labels: {data['genre_labels'].shape}, \n\tDancer Labels: {data['dancer_labels'].shape}"
        )
        logging.info("Pos: {}, Q: {}".format(
            data["pos"].shape, data["q"].shape))

        pose_input = self.process_dataset(data["pos"], data["q"])

        self.data = {
            "video_embeddings": data["video_embeddings"],
            "video_masks": data["video_masks"],
            "pose_estimations": data["pose_estimations"],
            "genre_labels": data["genre_labels"],
            "dancer_labels": data["dancer_labels"],
            "pose": pose_input,
            "filenames": data["filenames"],
            "wavs": data["wavs"],
        }

        assert (
            len(data["video_embeddings"])
            == len(data["pose_estimations"])
            == len(data["video_masks"])
            == len(data["genre_labels"])
            == len(data["dancer_labels"])
            == len(pose_input)
            == len(data["filenames"])
            == len(data["wavs"])
        )
        self.length = len(data["video_embeddings"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return (
            self.data["video_embeddings"][idx],
            self.data["video_masks"][idx],
            self.data["pose_estimations"][idx],
            self.data["genre_labels"][idx],
            self.data["dancer_labels"][idx],
            self.data["pose"][idx],
            feature,
            filename_,
            self.data["wavs"][idx],
        )

    def get_dancer_num(self):
        labels = self.data["dancer_labels"]
        if isinstance(labels, np.ndarray):
            return int(labels.max()) + 1
        return int(labels.max().item()) + 1

    def get_genre_num(self):
        labels = self.data["genre_labels"]
        if isinstance(labels, np.ndarray):
            return int(labels.max()) + 1
        return int(labels.max().item()) + 1

    def read_npy_files(self, file_path: str):
        return np.load(file_path, allow_pickle=True)  # (T, H, W, 3)

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
        return parse_aist_labels_from_name(filename)

    def read_label(self, filename):
        genre_code, dancer_code = self.parse_aist_labels(filename)

        if self.fixed_label_maps:
            # We expect these to already exist in the global mapping
            assert genre_code in self.genre2id, f"Unknown genre {genre_code}"
            assert dancer_code in self.dancer2id, f"Unknown dancer {dancer_code}"
            genre_id = self.genre2id[genre_code]
            dancer_id = self.dancer2id[dancer_code]
        else:
            # Build mapping on-the-fly (e.g. if you're using only one split somewhere)
            if genre_code not in self.genre2id:
                self.genre2id[genre_code] = len(self.genre2id)
            if dancer_code not in self.dancer2id:
                self.dancer2id[dancer_code] = len(self.dancer2id)
            genre_id = self.genre2id[genre_code]
            dancer_id = self.dancer2id[dancer_code]

        return genre_id, dancer_id

    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
            # basically (y, z) -> (-z, y), expressed as a rotation for readability
        )

        # do FK
        # batch x sequence x 24 x 3
        positions = smpl.forward(local_q, root_pos)
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(
            global_pose_vec_input)

        assert not torch.isnan(global_pose_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # # cut the dataset
        # if self.data_len > 0:
        #     global_pose_vec_input = global_pose_vec_input[: self.data_len]

        global_pose_vec_input = global_pose_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input

    def load_aistpp(self):
        # open data path
        split_root = os.path.join(
            self.data_path, "train" if self.train else "test")

        # Structure:
        # data
        #   |- train
        #   |    |- videos
        #   |    |- pose_estimation

        video_embedding_path = os.path.join(
            split_root, "video_features_sliced")
        video_mask_path = os.path.join(split_root, "video_mask_sliced")
        pose_estimation_path = os.path.join(
            split_root, "pose_estimation_sliced")
        motion_path = os.path.join(split_root, "motions_sliced")
        wav_path = os.path.join(split_root, "wavs_sliced")
        sound_path = os.path.join(split_root, f"{self.feature_type}_feats")

        # sort motions and sounds
        video_embeddings = sorted(
            glob.glob(os.path.join(video_embedding_path, "*.npy"))
        )
        video_masks = sorted(glob.glob(os.path.join(video_mask_path, "*.npy")))
        pose_estimations = sorted(
            glob.glob(os.path.join(pose_estimation_path, "*.json"))
        )

        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        assert (
            len(video_embeddings)
            == len(pose_estimations)
            == len(video_masks)
            == len(motions)
            == len(features)
            == len(wavs)
        ), f"Count mismatch: video_embeddings={len(video_embeddings)}, \
            pose_estimations={len(pose_estimations)}, \
            video_masks={len(video_masks)}, \
            motions={len(motions)}, \
            features={len(features)}, \
            wavs={len(wavs)}"

        (
            all_video_embeddings,
            all_video_masks,
            all_pose_estimations,
            all_genre_labels,
            all_dancer_labels,
            all_pos,
            all_q,
            all_wavs,
            all_names,
        ) = ([], [], [], [], [], [], [], [], [])

        for (
            video_embedding_filename,
            video_mask_filename,
            pose_est_filename,
            motion,
            feature,
            wav,
        ) in tqdm(
            zip(
                video_embeddings,
                video_masks,
                pose_estimations,
                motions,
                features,
                wavs,
            ),
            total=len(video_embeddings),
            desc="Loading data",
        ):
            v_name = os.path.splitext(
                os.path.basename(video_embedding_filename))[0]
            p_name = os.path.splitext(os.path.basename(pose_est_filename))[0]
            mask_name = os.path.splitext(
                os.path.basename(video_mask_filename))[0]
            m_name = os.path.splitext(os.path.basename(motion))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]

            assert (
                v_name == p_name == mask_name == m_name == f_name == w_name
            ), "Name mismatch {}, {}, {}, {}, {}, {}".format(
                v_name, p_name, mask_name, m_name, f_name, w_name
            )

            video_embedding = self.read_npy_files(video_embedding_filename)
            all_video_embeddings.append(video_embedding)

            video_mask = self.read_npy_files(video_mask_filename)
            all_video_masks.append(video_mask)

            pose_est = self.read_pose_estimation(
                pose_est_filename, vid_size=None)
            all_pose_estimations.append(pose_est)

            genre_id, dancer_id = self.read_label(v_name)
            all_genre_labels.append(genre_id)
            all_dancer_labels.append(dancer_id)

            data = pickle.load(open(motion, "rb"))
            # print(data.keys())
            pos = data["smpl_trans"]
            q = data["smpl_poses"]
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(feature)
            all_wavs.append(wav)

        all_video_embeddings = np.array(
            all_video_embeddings)  # N x T x H x W x 3
        all_video_masks = np.array(all_video_masks)  # N x T x H' x W'
        all_pose_estimations = np.array(all_pose_estimations)  # N x T x 17 x 3
        all_genre_labels = np.array(all_genre_labels)  # N
        all_dancer_labels = np.array(all_dancer_labels)  # N
        all_pos = np.array(all_pos)  # N x seq x 3
        all_q = np.array(all_q)  # N x seq x (joint * 3)
        # downsample the motions to the data fps
        print(all_pos.shape)
        # all_pos = all_pos[:, :: self.data_stride, :]
        # all_q = all_q[:, :: self.data_stride, :]

        data = {
            "video_embeddings": all_video_embeddings,
            "video_masks": all_video_masks,
            "pose_estimations": all_pose_estimations,
            "genre_labels": all_genre_labels,
            "dancer_labels": all_dancer_labels,
            "pos": all_pos,
            "q": all_q,
            "filenames": all_names,
            "wavs": all_wavs,
        }

        print("Video Embeddings Shape:", all_video_embeddings.shape)
        print("Pose Estimations Shape:", all_pose_estimations.shape)
        print("Video Masks Shape:", all_video_masks.shape)
        print("Genre Labels Shape:", all_genre_labels.shape)
        print("Dancer Labels Shape:", all_dancer_labels.shape)
        print("pos Shape:", all_pos.shape)
        print("q Shape:", all_q.shape)
        print("Number of files:", len(all_names))
        print("wavs Shape:", len(all_wavs))

        return data
