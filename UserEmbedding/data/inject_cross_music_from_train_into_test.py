#!/usr/bin/env python3
import os
import shutil
import random
from collections import defaultdict

# ================== CONFIG ==================
DATA_ROOT = "/raid/ltnghia02/vyttt/dance_gen_v2/UserEmbedding/data"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

TARGET_FOLDERS = [
    "video_mask_sliced",
    "pose_estimation_sliced",
    "video_embedding_sliced",
]

RANDOM_SEED = 42  # để reproducible
# ============================================


def parse_dancer_and_music(basename):
    """
    Expect format like:
      gJB_sFM_c01_d09_mJB1_ch21_slice8
    parts[3] -> dancer id (d09)
    parts[4] -> music id  (mJB1)
    """
    parts = basename.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {basename}")
    dancer = parts[3]
    music = parts[4]
    return dancer, music


def backup_folder(src, dst):
    if os.path.exists(dst):
        raise RuntimeError(f"Backup folder already exists: {dst}")
    print(f"[BACKUP] {src} -> {dst}")
    shutil.copytree(src, dst)


def build_basenames_and_exts(folder):
    """
    Trả về:
      basenames: list basenames (không extension)
      ext_map: {basename: extension}
    """
    basenames = []
    ext_map = {}

    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        base, ext = os.path.splitext(fname)
        if not ext:
            continue
        basenames.append(base)
        if base in ext_map and ext_map[base] != ext:
            raise RuntimeError(
                f"Multiple extensions for base {base} in {folder}: "
                f"{ext_map[base]} vs {ext}"
            )
        ext_map[base] = ext

    basenames = sorted(set(basenames))
    return basenames, ext_map


def build_train_index(train_dir):
    """
    Từ train_dir, build:
      basenames_train: list basenames
      ext_train: {base: ext}
      dancer_train: {base: dancer}
      music_train: {base: music}
      donors_by_dancer: {dancer: [bases]}
    """
    basenames_train, ext_train = build_basenames_and_exts(train_dir)

    dancer_train = {}
    music_train = {}
    donors_by_dancer = defaultdict(list)

    for b in basenames_train:
        dancer, music = parse_dancer_and_music(b)
        dancer_train[b] = dancer
        music_train[b] = music
        donors_by_dancer[dancer].append(b)

    return basenames_train, ext_train, dancer_train, music_train, donors_by_dancer


def process_folder(folder_name):
    print("\n==============================")
    print(f"Processing folder: {folder_name}")
    print("==============================")

    test_dir = os.path.join(TEST_ROOT, folder_name)
    train_dir = os.path.join(TRAIN_ROOT, f"{folder_name}")
    backup_dir = os.path.join(TEST_ROOT, folder_name + "_original")

    if not os.path.isdir(test_dir):
        raise RuntimeError(f"Test folder not found: {test_dir}")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Train folder not found: {train_dir}")

    # 1. Backup test folder
    backup_folder(test_dir, backup_dir)

    # 2. Build index từ train
    (
        basenames_train,
        ext_train,
        dancer_train,
        music_train,
        donors_by_dancer,
    ) = build_train_index(train_dir)
    print(f"Train donors in {folder_name}: {len(basenames_train)} files")

    # 3. Lấy danh sách file test (sẽ bị overwrite nội dung)
    basenames_test, ext_test = build_basenames_and_exts(test_dir)
    print(f"Test files in {folder_name}: {len(basenames_test)} files")

    changed = 0
    unchanged = 0
    no_ext_match = 0

    for b in basenames_test:
        dancer_test, music_test = parse_dancer_and_music(b)
        ext_t = ext_test[b]

        donor_candidates = []
        for donor_base in donors_by_dancer.get(dancer_test, []):
            # cùng dancer, khác music
            if music_train[donor_base] == music_test:
                continue
            # đảm bảo extension match
            ext_src = ext_train[donor_base]
            if ext_src != ext_t:
                continue
            donor_candidates.append(donor_base)

        if donor_candidates:
            # chọn random 1 donor
            donor = random.choice(donor_candidates)
            src_path = os.path.join(train_dir, donor + ext_train[donor])
            dst_path = os.path.join(test_dir, b + ext_t)
            
            print(f"{src_path} -> {dst_path}")

            if not os.path.exists(src_path):
                raise RuntimeError(f"Donor file not found: {src_path}")

            shutil.copy2(src_path, dst_path)
            changed += 1
        else:
            # không tìm được donor phù hợp → copy lại từ backup (giữ nguyên nội dung)
            src_path = os.path.join(backup_dir, b + ext_t)
            dst_path = os.path.join(test_dir, b + ext_t)

            if not os.path.exists(src_path):
                # nếu backup cũng không có thì báo lỗi
                raise RuntimeError(f"No donor and no original for: {b}{ext_t}")

            shutil.copy2(src_path, dst_path)
            unchanged += 1

            # check nếu có donor dancer nhưng ext mismatch để debug
            for donor_base in donors_by_dancer.get(dancer_test, []):
                if music_train[donor_base] != music_test and ext_train[donor_base] != ext_t:
                    no_ext_match += 1
                    break

    print(f"Done folder {folder_name}:")
    print(f"  files replaced by donor from train (same dancer, diff music): {changed}")
    print(f"  files kept original (no suitable donor):                    {unchanged}")
    if no_ext_match > 0:
        print(f"  (note: some dancers had donors but extension mismatch) ")


def main():
    random.seed(RANDOM_SEED)

    for folder in TARGET_FOLDERS:
        process_folder(folder)

    print("\nAll done. Only test/{video_mask_sliced, pose_estimation_sliced, "
          "video_embedding_sliced} were modified.")
    print("Backups available as *_original in test/.")


if __name__ == "__main__":
    main()
