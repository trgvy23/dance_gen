import numpy as np
import pickle
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json

# kinetic, manual
import os
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import argparse

import matplotlib.pyplot as plt
import librosa


def get_mb(music_root, key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        # print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict["music_array"])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict["music_array"])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]

        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')

        # print(len(beats))
        return beat_axis


def get_music_beat_fromwav(fpath, length):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    # EPS = 1e-6
    data, _ = librosa.load(fpath, sr=SR)[:length]
    # print("loaded music data shape", data.shape)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    start_bpm = librosa.beat.tempo(y=data)[0]
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    return beat_idxs


def calc_db(keypoints, name=""):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(
        np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1
    )
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)


def calc_ba_score(pred_root, music_root):

    # gt_list = []
    ba_scores = []

    for pkl in os.listdir(pred_root):
        # print(pkl)
        if os.path.isdir(os.path.join(pred_root, pkl)):
            continue
        joint3d = np.load(os.path.join(pred_root, pkl), allow_pickle=True)["full_pose"]
        assert len(joint3d.shape) == 3

        # NOTE: idk LODGE do this
        joint3d = joint3d.reshape(joint3d.shape[0], 24 * 3).detach().cpu().numpy()
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 24))
        joint3d = joint3d.reshape(-1, 24, 3)

        dance_beats, length = calc_db(joint3d, pkl)
        music_beats = get_music_beat_fromwav(
            os.path.join(music_root, pkl.split(".")[0] + ".wav"), joint3d.shape[0]
        )

        ba_scores.append(BA(music_beats, dance_beats))

    return np.mean(ba_scores)


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_root",
        type=str,
        default="/raid/ltnghia02/vyttt/dance_gen_v2/UserEmbedding/render/1799/motion_result/",
        help="Where to load saved motions",
    )
    parser.add_argument(
        "--music_root",
        type=str,
        default="wavs/",
        help="Where to load musics",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()

    print("Calculating and saving features")
    print(calc_ba_score(opt.pred_root, opt.music_root))
