import pprint


class DataID:
    EXPECTED_PARTS = 6

    def __init__(self, id: str):
        parts = id.split("_")
        if len(parts) != self.EXPECTED_PARTS:
            raise ValueError(f"Invalid ID format: {id}")

        self.raw_id = id
        (
            self.dance_genre,
            self.situation,
            self.camera_id,
            self.dancer_id,
            self.music_id,
            self.choreography_id,
        ) = parts

    def __repr__(self):
        data = {
            "raw_id": self.raw_id,
            "dance_genre": self.dance_genre,
            "situation": self.situation,
            "camera_id": self.camera_id,
            "dancer_id": self.dancer_id,
            "music_id": self.music_id,
            "choreography": self.choreography_id,
        }
        return pprint.pformat(data, indent=4)
    
    def _build_raw_id(self):
        self.raw_id = "_".join([
            self.dance_genre,
            self.situation,
            self.camera_id,
            self.dancer_id,
            self.music_id,
            self.choreography_id,
        ])
    
    def update_music_id(self, new_music_id):
        self.music_id = new_music_id
        self._build_raw_id()
