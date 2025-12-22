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


# def add_test_from_train(
#     train_data: list[DataID],
#     test_data: list[DataID],
#     K: int = 2,
#     max_iters: int = 1000,
#     include_all_dancers: bool = True,
# ) -> tuple:
#     """
#     Move whole music IDs from train -> test so every dancer has at least K distinct musics in test.
#     Returns (new_train, new_test, moved_music_ids).
#     Uses a greedy weighted set-cover approximation (benefit/cost).
#     """

#     # 1) current distinct music by dancer in test
#     dancer_test_musics = defaultdict(set)
#     for d in test_data:
#         dancer_test_musics[d.dancer_id].add(d.music_id)

#     # 2) dancer deficits (how many more distinct musics needed)
#     dancer_need = {}
#     dancers = set()
#     if include_all_dancers:
#         for d in train_data + test_data:
#             dancers.add(d.dancer_id)
#     else:
#         for d in test_data:
#             dancers.add(d.dancer_id)
#     for dancer in dancers:
#         have = len(dancer_test_musics.get(dancer, set()))
#         dancer_need[dancer] = max(0, K - have)

#     # quick check: if everyone already satisfied, return
#     if all(n == 0 for n in dancer_need.values()):
#         return train_data, test_data, set()

#     # 3) Index: which dancers have which music in TRAIN (unique music per dancer)
#     music_to_dancers_in_train = defaultdict(set)
#     music_train_examples_count = Counter()
#     dancer_music_in_train = defaultdict(set)

#     for rec in train_data:
#         music_train_examples_count[rec.music_id] += 1
#         music_to_dancers_in_train[rec.music_id].add(rec.dancer_id)
#         dancer_music_in_train[rec.dancer_id].add(rec.music_id)

#     # Exclude music IDs already present in test (we want test-exclusive musics)
#     music_in_test = {d.music_id for d in test_data}
#     candidate_music_ids = {m for m in music_to_dancers_in_train.keys() if m not in music_in_test}

#     # 4) Feasibility check: each dancer must have enough distinct candidate musics in train+test
#     impossible = []
#     for dancer in dancers:
#         total_distinct_available = len(set(dancer_test_musics.get(dancer, set())) | dancer_music_in_train.get(dancer, set()))
#         if total_distinct_available < K:
#             impossible.append(dancer)
#     if impossible:
#         raise ValueError(f"Cannot satisfy K={K} for dancers (not enough distinct musics): {impossible}")

#     # 5) Greedy selection loop
#     moved_music_ids = set()
#     # We'll update dancer_need as we move music ids
#     iters = 0
#     while True:
#         iters += 1
#         if iters > max_iters:
#             raise RuntimeError("Reached max iterations in greedy selection")

#         # compute per-music benefit (how many dancer deficits it would reduce)
#         best_score = -1.0
#         best_music = None
#         best_benefit = 0

#         for m in list(candidate_music_ids):
#             # benefit = number of dancers this music would contribute to (but capped by dancer_need)
#             benefit = 0
#             for dancer in music_to_dancers_in_train[m]:
#                 if dancer_need.get(dancer, 0) > 0 and m not in dancer_test_musics.get(dancer, set()):
#                     # moving m gives this dancer one additional distinct music in test
#                     benefit += 1

#             if benefit == 0:
#                 continue

#             cost = music_train_examples_count[m]  # how many train examples we'd remove
#             # scoring: benefit per unit cost (higher is better); break ties by larger benefit
#             score = benefit / cost

#             if score > best_score or (score == best_score and benefit > best_benefit):
#                 best_score = score
#                 best_music = m
#                 best_benefit = benefit

#         if best_music is None:
#             # no candidate helps remaining deficits -> should not happen due to feasibility check
#             break

#         # select best_music: move all train examples with this music into test
#         moved_music_ids.add(best_music)
#         # remove from candidate set
#         candidate_music_ids.remove(best_music)

#         # update dancer_test_musics and dancer_need
#         for dancer in music_to_dancers_in_train[best_music]:
#             if dancer_need.get(dancer, 0) > 0 and best_music not in dancer_test_musics.get(dancer, set()):
#                 dancer_test_musics[dancer].add(best_music)
#                 dancer_need[dancer] -= 1

#         # stop if everyone satisfied
#         if all(n == 0 for n in dancer_need.values()):
#             break

#     # 6) Build new train/test lists (move all recs whose music_id in moved_music_ids)
#     new_train = [rec for rec in train_data if rec.music_id not in moved_music_ids]
#     moved_recs = [rec for rec in train_data if rec.music_id in moved_music_ids]
#     new_test = test_data + moved_recs

#     return new_train, new_test

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
