import typing


class TrackedFace(object):

    _id_counter = 0
    _confirm_after = 1
    _remove_after = 3

    def __init__(self, bbox: typing.List[int], prob: float):
        TrackedFace._id_counter += 1
        self._id: int = self._id_counter
        self._bbox: typing.List[int] = bbox
        self._prob: float = prob
        self._just_detected: bool = True
        self._confirmed: bool = self._confirm_after == 0
        self._confirm_count: int = 0 if self._confirmed else 1
        self._removed: bool = False
        self._remove_count: int = 0
        self._class_id = None
        self.metadata = {}

    def update(self, bbox: typing.List[int], detected: bool = False, prob: float = None):
        if self._removed:
            raise RuntimeError("trying to update removed track")
        self._bbox = bbox
        self._just_detected = detected
        if detected:
            if prob is None:
                raise RuntimeError("detected must be set with prob")
            self._remove_count = 0
            self._prob = prob
            if not self._confirmed:
                self._confirm_count += 1
                if self._confirm_count > self._confirm_after:
                    self._confirm_count = 0
                    self._confirmed = True

    def set_not_detected(self):
        if not self._confirmed:
            self._removed = True
        else:
            self._remove_count += 1
            if self._remove_count >= self._remove_after:
                self._removed = True

    def set_class_id(self, class_id):
        if self._class_id is not None:
            raise RuntimeError("unable to reassign class")
        self._class_id = class_id

    @property
    def id(self) -> int:
        return self._id

    @property
    def bbox(self) -> typing.List[int]:
        return self._bbox

    @property
    def prob(self) -> float:
        return self._prob

    @property
    def class_id(self) -> int:
        return self._class_id

    @property
    def just_detected(self) -> bool:
        return self._just_detected

    @property
    def remove_count(self) -> int:
        return self._remove_count

    @property
    def removed(self) -> bool:
        return self._removed

    @property
    def to_remove(self) -> bool:
        return self._remove_count > 0

    @property
    def confirm_count(self) -> int:
        return self._confirm_count

    @property
    def confirmed(self) -> bool:
        return self._confirmed
