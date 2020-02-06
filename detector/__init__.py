import abc

import typing
import numpy as np

from tools import bbox


class Detector(object):
    def __init__(self,
                threshold: float = 0.5,
                split_counts: str = "2",
                face_min_size: int = 10,
    ):
        self._threshold = threshold
        try:
            self._split_counts: typing.List[int] = sorted(
                [int(s) for s in split_counts.split(",")]
            )
        except ValueError:
            self._split_counts: typing.List[int] = []
        self._face_min_area = face_min_size ** 2

    def detect(
        self, bgr_frame: np.ndarray
    ) -> (typing.List[typing.List[int]], typing.List[float]):
        boxes = self._detect_faces_split(bgr_frame)

        if len(self._split_counts) > 0:

            def add_box(b):
                for i, b0 in enumerate(boxes):
                    if bbox.box_intersection(b0, b) > 0.3:
                        # set the largest proba to existing box
                        boxes[i][4] = max(b0[4], b[4])
                        return
                boxes.resize((boxes.shape[0] + 1, boxes.shape[1]), refcheck=False)
                boxes[-1] = b

            for split_count in self._split_counts:
                size_multiplier = 2.0 / (split_count + 1)
                xstep = int(bgr_frame.shape[1] / (split_count + 1))
                ystep = int(bgr_frame.shape[0] / (split_count + 1))

                xlimit = int(np.ceil(bgr_frame.shape[1] * (1 - size_multiplier)))
                ylimit = int(np.ceil(bgr_frame.shape[0] * (1 - size_multiplier)))
                for x in range(0, xlimit, xstep):
                    for y in range(0, ylimit, ystep):
                        y_border = min(
                            bgr_frame.shape[0],
                            int(np.ceil(y + bgr_frame.shape[0] * size_multiplier)),
                        )
                        x_border = min(
                            bgr_frame.shape[1],
                            int(np.ceil(x + bgr_frame.shape[1] * size_multiplier)),
                        )
                        crop = bgr_frame[y:y_border, x:x_border, :]

                        box_candidates = self._detect_faces_split(crop, (x, y))

                        for b in box_candidates:
                            add_box(b)

        return boxes[:, :4].astype(int), boxes[:, 4]

    @abc.abstractmethod
    def _detect_faces_split(self, bgr_frame: np.ndarray, offset=(0, 0)):
        raise NotImplementedError
