def get_dancer_id(data: str) -> str:
    return data.split("_")[3]


def get_music_id(data: str) -> str:
    return data.split("_")[4]
