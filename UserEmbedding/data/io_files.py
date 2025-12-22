from typing import Optional
import os
import shutil

from data_id import DataID
from split import split_from_files, add_test_from_train
from shuffle import shuffle, import_shuffle_map


def load_data_ids(input_path: str) -> list[DataID]:
    data_list = []
    with open(input_path, "r") as f:
        for line in f:
            raw_id = line.strip()
            if raw_id:
                data_list.append(DataID(raw_id))

    return data_list


def save_data_ids(data_list: list[DataID], output_path: str):
    with open(output_path, "w") as f:
        for data in data_list:
            f.write(f"{data.raw_id}\n")


def extract_unique_music_ids(data_list: list[DataID]) -> set[str]:
    return {data.music_id for data in data_list}


def recreate_empty_dir(path: str):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(path, exist_ok=True)


def rename_filename_with_shuffle(
    filename: str,
    shuffle_map: dict[str, str],
):
    stem, ext = os.path.splitext(filename)
    parts = stem.split("_")

    if len(parts) < 6:
        return filename  # untouched

    music_id = parts[4]
    if music_id not in shuffle_map:
        return filename

    parts[4] = shuffle_map[music_id]
    return "_".join(parts) + ext


def copy_dataset_with_shuffle(
    src_root: str,
    dst_root: str,
    shuffle_map: dict[str, str],
    extensions: Optional[set[str]] = None,
):
    recreate_empty_dir(dst_root)

    for root, _, files in os.walk(src_root):
        rel_path = os.path.relpath(root, src_root)
        dst_dir = dst_root if rel_path == "." else os.path.join(dst_root, rel_path)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in files:
            if extensions and not any(fname.endswith(ext) for ext in extensions):
                continue

            new_name = rename_filename_with_shuffle(fname, shuffle_map)

            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_dir, new_name)

            shutil.copy2(src_path, dst_path)


def process_data(
    train_data_file_path: str,
    test_data_file_path: str,
    new_train_data_file_path: str,
    new_test_data_file_path: str,
    original_dataset_path: str,
    new_dataset_path: str,
    shuffling_file_path: str,
    do_shuffle: bool = False,
):
    train_data = load_data_ids(train_data_file_path)
    test_data = load_data_ids(test_data_file_path)
    ignore_data = load_data_ids(ignore_data_file_path)
    
    train_data, test_data = split_from_files(train_data, test_data, ignore_data)
    if do_shuffle or not os.path.isfile(shuffling_file_path):
        train_data, test_data = shuffle(train_data, test_data, shuffling_file_path)
    # train_data, test_data = add_test_from_train(train_data, test_data)
    train_data, test_data = add_test_from_train(train_data, test_data)

    save_data_ids(train_data, new_train_data_file_path)
    save_data_ids(test_data, new_test_data_file_path)
    # rename file
    shuffle_map = import_shuffle_map(shuffling_file_path)

    copy_dataset_with_shuffle(
        src_root = original_dataset_path,
        dst_root = new_dataset_path,
        shuffle_map = shuffle_map,
        extensions = {".mp4", ".json", ".wav", ".pkl", ".npy"},
    )

    
if __name__ == "__main__":
    # input paths
    train_data_file_path = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/crossmodal_train.txt"
    test_data_file_path = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/crossmodal_test.txt"
    ignore_data_file_path = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/ignore_list.txt"

    # output paths
    new_train_data_file_path = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/new_crossmodel_train_test_only.txt"
    new_test_data_file_path = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/new_crossmodel_test_test_only.txt"
    shuffling_file_path = "/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/shuffling_map.txt"

    src_dataset_path = "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp/"
    dst_dataset_path = "/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp_shuffle/"

    process_data(
        train_data_file_path,
        test_data_file_path,
        new_train_data_file_path,
        new_test_data_file_path,
        src_dataset_path,
        dst_dataset_path,
        shuffling_file_path
    )
