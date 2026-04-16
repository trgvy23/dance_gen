import os
import shutil
from tqdm import tqdm

# ===== HARD-CODED PATHS =====
LIST_FILE = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/test.txt"

SRC_AUDIO_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/wavs"
SRC_VIDEO_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/video"
SRC_MOTION_DIR = "/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp/motions"

DST_AUDIO_DIR = "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp_processed/test/wavs"
DST_VIDEO_DIR = "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp_processed/test/video"
DST_MOTION_DIR = "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp_processed/test/motions"
# ============================

os.makedirs(DST_AUDIO_DIR, exist_ok=True)
os.makedirs(DST_VIDEO_DIR, exist_ok=True)
os.makedirs(DST_MOTION_DIR, exist_ok=True)


def read_ids(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def copy_if_exists(src, dst):
    if not os.path.isfile(src):
        return False
    shutil.copy2(src, dst)
    return True


def main():
    ids = read_ids(LIST_FILE)

    missing_wav = 0
    missing_mp4 = 0
    missing_pkl = 0
    missing_sample = 0

    for name in tqdm(ids, desc="Copying samples", unit="sample"):
        ok = True

        if not copy_if_exists(
            os.path.join(SRC_AUDIO_DIR, name + ".wav"),
            os.path.join(DST_AUDIO_DIR, name + ".wav"),
        ):
            missing_wav += 1
            ok = False

        if not copy_if_exists(
            os.path.join(SRC_VIDEO_DIR, name + ".mp4"),
            os.path.join(DST_VIDEO_DIR, name + ".mp4"),
        ):
            missing_mp4 += 1
            ok = False

        if not copy_if_exists(
            os.path.join(SRC_MOTION_DIR, name + ".pkl"),
            os.path.join(DST_MOTION_DIR, name + ".pkl"),
        ):
            missing_pkl += 1
            ok = False

        if not ok:
            missing_sample += 1

    print("\nDone ✅")
    print(f"Missing samples : {missing_sample}/{len(ids)}")
    print(f"Missing wav     : {missing_wav}")
    print(f"Missing mp4     : {missing_mp4}")
    print(f"Missing pkl     : {missing_pkl}")


if __name__ == "__main__":
    main()
