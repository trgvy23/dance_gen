import numpy as np
import copy
import re
import os
import glob


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
    ratio = np.random.uniform(
        low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def parse_aist_labels_from_name(filename: str):
    """
    filename example: 'gBR_sBM_cAll_d04_mBR0_ch02'
    returns: ('gBR', 'd04')
    """
    m = re.match(r"^(g[A-Z]{2})_.*_(d\d+)_m", filename)
    assert m, f"Bad AIST++ name: {filename}"
    genre_code = m.group(1)  # 'gBR'
    dancer_code = m.group(2)  # 'd04'
    return genre_code, dancer_code


def build_label_mappings(data_path: str, splits=("train", "test")):
    """
    Scan all splits and build a single consistent genre2id, dancer2id mapping.
    """
    genre2id = {}
    dancer2id = {}

    print(f"Building label mappings for data_path: {data_path}")

    for split in splits:
        split_root = os.path.join(data_path, split)
        video_embedding_path = os.path.join(
            split_root, "video_features_sliced")
        print(f"Scanning {split} split in {
              video_embedding_path} for label mappings...")
        video_embeddings = sorted(
            glob.glob(os.path.join(video_embedding_path, "*.npy"))
        )

        for fpath in video_embeddings:
            v_name = os.path.splitext(os.path.basename(fpath))[0]
            genre_code, dancer_code = parse_aist_labels_from_name(v_name)

            if genre_code not in genre2id:
                genre2id[genre_code] = len(genre2id)
            if dancer_code not in dancer2id:
                dancer2id[dancer_code] = len(dancer2id)

    return genre2id, dancer2id
