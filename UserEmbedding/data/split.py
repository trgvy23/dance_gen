from collections import defaultdict, Counter

from data_id import DataID


def extract_unique_music_ids(data_list: list[DataID]) -> set[str]:
    return {data.music_id for data in data_list}


def add_test_data(train_data: list[DataID], test_data: list[DataID], ignore_data: list[DataID]) -> list[DataID]:
    """
    Rule:
    - If a music_id appears only in ignore and not in train,
      all its data goes to test.
    """
    train_music_ids = extract_unique_music_ids(train_data)
    ignore_music_ids = extract_unique_music_ids(ignore_data)

    new_test_music_ids = ignore_music_ids - train_music_ids

    additional_test_data = [
        data for data in ignore_data
        if data.music_id in new_test_music_ids
    ]

    return test_data + additional_test_data


def split_from_files(train_data_list: list[DataID], test_data_list: list[DataID], ignore_data_list: list[DataID]):
    test_data_list = add_test_data(train_data_list, test_data_list, ignore_data_list)
    return train_data_list, test_data_list


def add_test_from_train(
    train: list[DataID],
    test: list[DataID],
    min_test_per_dancer: int = 2
) -> tuple:

    train = list(train)
    test = list(test)

    # Helper to rebuild music -> items map
    def build_music_map(items):
        m = defaultdict(list)
        for x in items:
            m[x.music_id].append(x)
        return m

    # Step 1: mandatory move (music exclusivity)
    test_music = {x.music_id for x in test}
    mandatory = [x for x in train if x.music_id in test_music]

    for x in mandatory:
        train.remove(x)
        test.append(x)

    # Main loop
    while True:
        dancer_test_count = Counter(x.dancer_id for x in test)

        # All dancers currently in test must satisfy constraint
        deficits = {
            d: min_test_per_dancer - c
            for d, c in dancer_test_count.items()
            if c < min_test_per_dancer
        }

        if not deficits:
            break  # done

        music_map = build_music_map(train)

        best_music = None
        best_score = float("inf")
        best_benefit = 0

        # Evaluate all candidate music once
        for music_id, items in music_map.items():
            cost = len(items)
            benefit = 0

            # How much does this music help current deficits?
            for x in items:
                if x.dancer_id in deficits:
                    benefit += 1

            benefit = min(
                benefit,
                sum(deficits.values())
            )

            if benefit == 0:
                continue

            score = cost / benefit

            if score < best_score or (score == best_score and benefit > best_benefit):
                best_score = score
                best_music = music_id
                best_benefit = benefit

        if best_music is None:
            raise ValueError("Impossible to satisfy dancer constraints")

        # Move chosen music
        for x in list(music_map[best_music]):
            train.remove(x)
            test.append(x)

            moved = True
            break  # re-evaluate from scratch (important!)

        if not moved:
            raise RuntimeError("Stuck while fixing test constraints")

    return train, test
