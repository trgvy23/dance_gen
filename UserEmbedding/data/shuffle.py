import re
import random
from collections import defaultdict
from typing import Optional

from data_id import DataID


def extract_unique_music_ids(data_list: list[DataID]) -> set[str]:
    return {data.music_id for data in data_list}


def get_music_genre(music_id: str) -> str:
    return re.sub(r"\d+$", "", music_id)


def build_genre_shuffle_map(music_ids: list[str]) -> dict[str, str]:
    genre_groups = defaultdict(list)
    for music_id in music_ids:
        genre_groups[get_music_genre(music_id)].append(music_id)

    shuffle_map = {}

    for ids in genre_groups.values():
        if len(ids) <= 1:
            shuffle_map[ids[0]] = ids[0]
            continue

        shuffled = ids[:]
        is_shuffled = False
        while not is_shuffled:
            random.shuffle(shuffled)
            is_shuffled = all(a != b for a, b in zip(ids, shuffled))

        shuffle_map.update(dict(zip(ids, shuffled)))

    return shuffle_map


def _export_shuffle_map(shuffle_map: dict[str, str], path: str) -> None:
    with open(path, "w") as f:
        f.write("old_music_id -> new_music_id\n")
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
    export_map_path: Optional[str] = None
) -> tuple:
    music_ids = {data.music_id for data in train_data + test_data}
    # pprint.pprint(music_ids)
    shuffle_music_ids_map = build_genre_shuffle_map(music_ids)
    # pprint.pprint(shuffle_music_ids_map)

    # print("before:")
    # print(f"\ttrain: {train_data[:5]}")
    # print(f"\ttest: {test_data[:5]}")

    for data in train_data + test_data:
        data.update_music_id(shuffle_music_ids_map[data.music_id])

    # print("after:")
    # print(f"\ttrain: {train_data[:5]}")
    # print(f"\ttest: {test_data[:5]}")

    if export_map_path is not None:
        _export_shuffle_map(shuffle_music_ids_map, export_map_path)

    return train_data, test_data
