import re
import random
from collections import defaultdict
import os
import copy
import math
import shutil


from data_id import get_dancer_id


def build_genre_shuffle_map(data_list: list[str], max_try: int = 1_000_000) -> dict[str, str]:
    dancer_id = defaultdict(list)
    for data in data_list:
        dancer_id[get_dancer_id(data)].append(data)

    shuffle_map = defaultdict(str)
    for dancer_data_list in dancer_id.values():
        if len(dancer_data_list) < 2:
            new_data_list = list(dancer_data_list)
        else:
            cnt = 1
            visited = []
            while True and cnt < min(max_try, math.factorial(len(dancer_data_list))):
                # print(f"Try {cnt}")
                new_data_list = copy.deepcopy(dancer_data_list)
                random.shuffle(new_data_list)
                sub_cnt = 1
                while new_data_list in visited and sub_cnt < min(max_try, math.factorial(len(dancer_data_list))):
                    # print(f"Sub try {sub_cnt}")
                    random.shuffle(new_data_list)
                    sub_cnt += 1
                visited.append(new_data_list)
                cnt += 1
                if len(visited) > math.factorial(len(dancer_data_list)) or all(a != b for a, b in zip(dancer_data_list, new_data_list)):
                    break
            if any(a == b for a, b in zip(dancer_data_list, new_data_list)):
                print("Cannot shuffle -> rotate the list")
                new_data_list = dancer_data_list[1:] + [dancer_data_list[0]]
        for i in range(len(dancer_data_list)):
            shuffle_map[dancer_data_list[i]] = new_data_list[i]

    return shuffle_map


def _recreate_empty_dir(path: str):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(path, exist_ok=True)


def _ensure_dir(p: str, create_new: bool = False) -> None:
    if create_new:
        _recreate_empty_dir(p)
    else:
        os.makedirs(p, exist_ok=True)


def rename_with_mapping_os(
    directory: str,
    stem_map: dict[str, str],
    tmp_suffix: str = "_tmp"
):
    # ---- STEP 1: add temporary suffix ----
    _ensure_dir(directory)
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(filename)
        if stem not in stem_map:
            continue

        tmp_path = path + tmp_suffix
        os.rename(path, tmp_path)

    # ---- STEP 2: rename to final names ----
    for filename in os.listdir(directory):
        if not filename.endswith(tmp_suffix):
            continue

        tmp_path = os.path.join(directory, filename)

        original_name = filename[:-len(tmp_suffix)]
        stem, ext = os.path.splitext(original_name)

        new_stem = stem_map[stem]
        final_name = new_stem + ext
        final_path = os.path.join(directory, final_name)

        if os.path.exists(final_path):
            raise FileExistsError(f"Target already exists: {final_path}")

        os.rename(tmp_path, final_path)


def _export_shuffle_map(shuffle_map: dict[str, str], path: str) -> None:
    with open(path, "w") as f:
        f.write("old_data -> new_data\n")
        for old_id, new_id in sorted(shuffle_map.items()):
            f.write(f"{old_id} -> {new_id}\n")


def import_shuffle_map(path: str) -> dict[str, str]:
    shuffle_map = {}
    with open(path, "r") as f:
        for line in f:
            if "->" not in line:
                continue
            old_id, new_id = line.strip().split("->")
            shuffle_map[old_id.strip()] = new_id.strip()
    return shuffle_map


def shuffle(
    train_data: list,
    test_data: list,
    export_map_path: str = "",
) -> tuple:
    shuffle_music_ids_map = build_genre_shuffle_map(
        train_data) | build_genre_shuffle_map(test_data)

    if export_map_path != "":
        _export_shuffle_map(shuffle_music_ids_map, export_map_path)

    return train_data, test_data, shuffle_music_ids_map
