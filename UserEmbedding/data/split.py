from collections import defaultdict, Counter
import random

from data_id import get_dancer_id, get_music_id


def _add_test_core(
    train: list[str],
    test: list[str],
    min_test_per_dancer: int = 2,
) -> tuple:
    train = list(train)
    test = list(test)

    def build_music_map(items):
        m = defaultdict(list)
        for x in items:
            m[get_music_id(x)].append(x)
        return m

    # Step 1: mandatory move (music exclusivity)
    test_music = {get_music_id(x) for x in test}
    mandatory = [x for x in train if get_music_id(x) in test_music]

    for x in mandatory:
        train.remove(x)
        test.append(x)

    # Main loop
    while True:
        dancer_test_count = Counter(get_dancer_id(x) for x in test)

        deficits = {
            d: min_test_per_dancer - c
            for d, c in dancer_test_count.items()
            if c < min_test_per_dancer
        }

        if not deficits:
            break

        music_map = build_music_map(train)

        best_music = None
        best_score = float("inf")
        best_benefit = 0

        for music_id, items in music_map.items():
            cost = len(items)
            benefit = sum(1 for x in items if get_dancer_id(x) in deficits)

            benefit = min(benefit, sum(deficits.values()))

            if benefit == 0:
                continue

            score = cost / benefit

            if score < best_score or (score == best_score and benefit > best_benefit):
                best_score = score
                best_music = music_id
                best_benefit = benefit

        if best_music is None:
            raise ValueError("Impossible to satisfy dancer constraints")

        # Move ONE music group, then re-evaluate
        for x in list(music_map[best_music]):
            train.remove(x)
            test.append(x)
        # important: re-loop

    return train, test


def extract_unique_music_ids(data_list: list[str]) -> set[str]:
    return {get_music_id(data) for data in data_list}


def add_test_data(
    train_data: list[str], test_data: list[str], ignore_data: list[str]
) -> list[str]:
    """
    Rule:
    - If a music_id appears only in ignore and not in train,
      all its data goes to test.
    """
    train_music_ids = extract_unique_music_ids(train_data)
    ignore_music_ids = extract_unique_music_ids(ignore_data)

    new_test_music_ids = ignore_music_ids - train_music_ids

    additional_test_data = [
        data for data in ignore_data if get_music_id(data) in new_test_music_ids
    ]

    return test_data + additional_test_data


def split_from_list_data(
    train_data_list: list[str],
    test_data_list: list[str],
    ignore_data_list: list[str] = [],
):
    test_data_list = add_test_data(train_data_list, test_data_list, ignore_data_list)
    test_data_list_set = set(test_data_list)
    train_data_list = [
        data for data in train_data_list if data not in test_data_list_set
    ]
    train_data_list, test_data_list, ignore_data_list = add_test_from_train(
        train_data_list, test_data_list
    )
    return train_data_list, test_data_list, ignore_data_list


def enforce_dancer_disjointness(
    train: list[str],
    test: list[str],
    ignore: list[str],
    min_test_per_dancer: int,
) -> tuple[list[str], list[str], list[str]]:
    test_dancers = {get_dancer_id(x) for x in test}
    dancer_test_count = Counter(get_dancer_id(x) for x in test)

    new_train = []
    new_test = list(test)
    new_ignore = list(ignore)

    for x in train:
        d = get_dancer_id(x)

        if d not in test_dancers:
            new_train.append(x)
            continue

        # dancer already in test → cannot stay in train
        if dancer_test_count[d] < min_test_per_dancer:
            new_test.append(x)
            dancer_test_count[d] += 1
        else:
            new_ignore.append(x)

    return new_train, new_test, new_ignore


def add_test_from_train(
    train: list[str],
    test: list[str],
    min_test_per_dancer: int = 2,
) -> tuple:
    original_train = list(train)
    original_test = list(test)

    # If test already has data, check if it satisfies the constraints
    if test and len(test) > 0:
        # 1. dancer count constraint
        dancer_count = Counter(get_dancer_id(x) for x in test)
        dancers_ok = all(c >= min_test_per_dancer for c in dancer_count.values())

        # 2. music disjointness
        train_music = {get_music_id(x) for x in train}
        test_music = {get_music_id(x) for x in test}
        music_ok = train_music.isdisjoint(test_music)

        # 3. dancer disjointness (NEW)
        train_dancers = {get_dancer_id(x) for x in train}
        test_dancers = {get_dancer_id(x) for x in test}
        dancer_ok = train_dancers.isdisjoint(test_dancers)

        if dancers_ok and music_ok and dancer_ok:
            return train, test, []

    # Otherwise: retry with different initial seeds
    candidates = list(range(len(train)))
    random.shuffle(candidates)

    ignore_data = []
    for idx in candidates:
        train_try = list(original_train)
        test_try = list(original_test)

        # Seed test with 1 random sample
        test_try.append(train_try.pop(idx))

        try:
            train_ok, test_ok = _add_test_core(train_try, test_try, min_test_per_dancer)

            train_ok, test_ok, ignore_data = enforce_dancer_disjointness(
                train_ok,
                test_ok,
                ignore_data,
                min_test_per_dancer,
            )

            return train_ok, test_ok, ignore_data
        except ValueError:
            # Failed due to constraints → retry
            continue

    raise ValueError("Unable to construct a valid test set from any initial seed")
