import os

import cv2
import typing
from ml_serving.drivers import driver
import numpy as np

from constants import GPU_ID
from constants import LOG_DIR
from tools import bbox
from tools import images
from tracker import re3_tracker

from sklearn.neighbors import KDTree

FACE_DETECTION_PATH = (
    "/opt/intel/openvino/deployment_tools/intel_models/"
    "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
)


class TrackedFace(object):

    _id_counter = 0
    _confirm_after = 1
    _remove_after = 5

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

    def update(
        self, bbox: typing.List[int], detected: bool = False, prob: float = None
    ):
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
    def class_id(self) -> float:
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


class FacesTracker(object):
    def __init__(
        self,
        re3_checkpoint_dir: str = os.path.join(
            os.path.dirname(__file__), "..", LOG_DIR, "checkpoints"
        ),
        face_detection_path: str = FACE_DETECTION_PATH,
        face_detection_threshold: float = .5,
        face_detection_split_counts: str = "2",
        face_detection_min_size: int = 10,
        facenet_path: str = os.path.join(os.path.dirname(__file__), "..", LOG_DIR, "facenet"),
        intersection_threshold: float = 0.2,
        detect_each: int = 10,
        gpu_id=GPU_ID,
        add_min=0.3,
        add_max=0.5,
    ):

        # intersection coef for identifying tracked and detected faces
        self._intersection_threshold: float = intersection_threshold

        self._re3_tracker: re3_tracker.Re3Tracker = re3_tracker.Re3Tracker(
            re3_checkpoint_dir, gpu_id=gpu_id
        )

        self._face_detect_driver: driver.ServingDriver = driver.load_driver(
            "openvino"
        )()
        self._face_detect_driver.load_model(face_detection_path)
        self._face_detect_threshold: float = face_detection_threshold
        try:
            self._face_detect_split_counts: typing.List[int] = sorted(
                [int(s) for s in face_detection_split_counts.split(",")]
            )
        except ValueError:
            self._face_detect_split_counts: typing.List[int] = []
        self._face_detect_min_area = face_detection_min_size ** 2

        self._facenet_driver: driver.ServingDriver = driver.load_driver("openvino")()
        self._facenet_driver.load_model(facenet_path)
        self._facenet_input_shape = list(self._facenet_driver.inputs.values())[0]
        self._facenet_input_name = list(self._facenet_driver.inputs)[0]
        self._facenet_output_name = list(self._facenet_driver.outputs)[0]

        self._detect_each: int = detect_each
        self._counter: int = -1

        self._tracked: typing.List[TrackedFace] = []

        self._kd_tree = None
        self._kd_embeddings = None
        self._kd_classes = []

        self._add_range = [add_min, add_max]

        self._l: typing.List[str] = []

    def track(self, frame: np.ndarray) -> typing.List[TrackedFace]:

        bgr_frame = frame[:, :, ::-1]

        self._counter += 1
        self._l = []

        self._log(f"frame {self._counter}")

        # existing tracks
        tracked_faces = self._track(bgr_frame, [t.id for t in self._tracked])

        if self._counter % self._detect_each > 0:
            for i, t in enumerate(tracked_faces):
                self._tracked[i].update(t.astype(int))
            return self._tracked

        # detected faces: bboxes and proabbilities
        detected_bboxes, detected_probs = self._detect_faces(bgr_frame)
        detected_track_ids = []

        for i, detected_bbox in enumerate(detected_bboxes):

            detected_prob = detected_probs[i]
            is_tracked = False
            track = None

            # update existing tracks with detected faces if found
            for t in self._tracked:
                if t.id not in detected_track_ids:
                    if (
                        bbox.box_intersection(detected_bbox, t.bbox)
                        > self._intersection_threshold
                    ):
                        is_tracked = True
                        t.update(detected_bbox, detected=True, prob=detected_prob)
                        track = t
                        break

            # add new tracks for faces not found in existing tracks
            if not is_tracked:
                track = TrackedFace(detected_bbox, detected_prob)
                self._tracked.append(track)

            # apply existing or new track updating
            if track is not None:
                detected_track_ids.append(track.id)
                self._track_add(bgr_frame, track.id, detected_bbox)

        # mark existing tracks as de
        for tr in self._tracked:
            if tr.id not in detected_track_ids:
                tr.set_not_detected()

        # clear removed tracks
        self._tracked = [t for t in self._tracked if not t.removed]

        # classify tracked faces
        classified_tracks = [
            t for t in self._tracked if t.confirmed and not t.to_remove
        ]
        if len(classified_tracks) > 0:
            confirmed_tracks_bboxes = [t.bbox for t in classified_tracks]
            embs = self._face_embeddings(frame, confirmed_tracks_bboxes)
            if self._kd_tree is None:
                class_ids = self._kd_init(embs)
                for i, t in enumerate(classified_tracks):
                    t.set_class_id(class_ids[i])
                self._log("init classifier with classes {}".format(class_ids))
            else:
                for i, emb in enumerate(embs):
                    dists, idxs = self._kd_tree.query(
                        emb.reshape([1, -1]), k=min(3, len(self._kd_classes))
                    )
                    dist = dists[0][0]
                    idx = idxs[0][0]
                    closest = [self._kd_classes[i] for i in idxs[0]]
                    track = classified_tracks[i]
                    if classified_tracks[i].class_id is None:
                        # TODO new class can be only one on one frame!
                        if dist <= self._add_range[1]:
                            existing_class = self._kd_classes[idx]
                            track.set_class_id(existing_class)
                            self._log(
                                "detected as existing class {} for track {}, distance {}, closest {}".format(
                                    existing_class, track.id, dist, closest
                                )
                            )
                            if dist > self._add_range[0]:
                                self._kd_add(emb, existing_class)
                                self._log(
                                    "added another emb for class {} from track {}, distance {}, closest {}".format(
                                        track.class_id, track.id, dist, closest
                                    )
                                )
                        else:
                            new_class = self._kd_add(emb)
                            track.set_class_id(new_class)
                            self._log(
                                "added new class {} from track {}, distance {}, closest {}".format(
                                    new_class, track.id, dist, closest
                                )
                            )
                    else:
                        if self._kd_classes[idx] != track.class_id:
                            self._log(
                                "detected as class {}, but tracked as {}, closest: {}, dists: {}".format(
                                    self._kd_classes[idx],
                                    track.class_id,
                                    closest,
                                    dists,
                                )
                            )
                        # confirm with new embedding
                        if dist > self._add_range[1]:
                            self._log(
                                "added new emb for existing class {} from track {}, distance {}, closest {}".format(
                                    track.class_id, track.id, dist, closest
                                )
                            )
                            self._kd_add(emb, track.class_id)

        return self._tracked

    def _track(self, bgr_frame: np.ndarray, indexes: [int]):
        if len(indexes) == 0:
            return []
        if len(indexes) == 1:
            return [self._re3_tracker.track(f"{indexes[0]}", bgr_frame)]
        return self._re3_tracker.multi_track([f"{i}" for i in indexes], bgr_frame)

    def _track_add(self, bgr_frame: np.ndarray, index: int, bbox: [int]):
        self._re3_tracker.track(f"{index}", bgr_frame, bbox)

    def _detect_faces(self, bgr_frame: np.ndarray):
        boxes = self._detect_faces_split(bgr_frame)

        if len(self._face_detect_split_counts) > 0:
            def add_box(b):
                for i, b0 in enumerate(boxes):
                    if bbox.box_intersection(b0, b) > 0.3:
                        # set the largest proba to existing box
                        boxes[i][4] = max(b0[4], b[4])
                        return
                boxes.resize((boxes.shape[0] + 1, boxes.shape[1]), refcheck=False)
                boxes[-1] = b

            for split_count in self._face_detect_split_counts:
                size_multiplier = 2. / (split_count + 1)
                xstep = int(bgr_frame.shape[1] / (split_count + 1))
                ystep = int(bgr_frame.shape[0] / (split_count + 1))

                xlimit = int(np.ceil(bgr_frame.shape[1] * (1 - size_multiplier)))
                ylimit = int(np.ceil(bgr_frame.shape[0] * (1 - size_multiplier)))
                for x in range(0, xlimit, xstep):
                    for y in range(0, ylimit, ystep):
                        y_border = min(bgr_frame.shape[0], int(np.ceil(y + bgr_frame.shape[0] * size_multiplier)))
                        x_border = min(bgr_frame.shape[1], int(np.ceil(x + bgr_frame.shape[1] * size_multiplier)))
                        crop = bgr_frame[y:y_border, x:x_border, :]

                        box_candidates = self._detect_faces_split(crop, (x, y))

                        for b in box_candidates:
                            add_box(b)

        return boxes[:, :4].astype(int), boxes[:, 4]

    def _detect_faces_split(self, bgr_frame: np.ndarray, offset=(0, 0)):

        drv = self._face_detect_driver
        # Get boxes shaped [N, 5]:
        # xmin, ymin, xmax, ymax, confidence
        input_name, input_shape = list(drv.inputs.items())[0]
        output_name = list(drv.outputs)[0]
        inference_frame = cv2.resize(
            bgr_frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA
        )
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
        outputs = drv.predict({input_name: inference_frame})
        output = outputs[output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > self._face_detect_threshold]
        # Extract 5 values
        boxes = bboxes_raw[:, 3:7]
        confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
        boxes = np.concatenate((boxes, confidence), axis=1)
        # Assign confidence to 4th
        # boxes[:, 4] = bboxes_raw[:, 2]
        boxes[:, 0] = boxes[:, 0] * bgr_frame.shape[1] + offset[0]
        boxes[:, 2] = boxes[:, 2] * bgr_frame.shape[1] + offset[0]
        boxes[:, 1] = boxes[:, 1] * bgr_frame.shape[0] + offset[1]
        boxes[:, 3] = boxes[:, 3] * bgr_frame.shape[0] + offset[1]
        if boxes is not None:
            boxes = boxes[(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) >= self._face_detect_min_area]
        return boxes

    def _face_embeddings(
        self, frame: np.ndarray, source: typing.List[typing.List[int]]
    ):
        face_images = images.get_images(
            frame,
            np.array([s[:4] for s in source]),
            normalization=images.DEFAULT_NORMALIZATION,
        )
        embeddings = []
        for face_img in face_images:
            face_img = face_img.transpose([2, 0, 1]).reshape(self._facenet_input_shape)
            outputs = self._facenet_driver.predict({self._facenet_input_name: face_img})
            output = outputs[self._facenet_output_name]
            embeddings.append(output.reshape([-1]))
        return (np.asarray(embeddings) + 1.0) / 2

    def _kd_init(self, embs: np.ndarray) -> typing.List:
        self._kd_embeddings = embs
        self._kd_classes = [i + 1 for i in range(len(embs))]
        self._kd_tree = KDTree(self._kd_embeddings, metric="euclidean")
        return self._kd_classes

    def _kd_add(self, emb: np.ndarray, class_id=None):
        self._kd_embeddings = np.concatenate(
            (self._kd_embeddings, emb.reshape([1, -1]))
        )
        if class_id is None:
            class_id = max(self._kd_classes) + 1
        self._kd_classes.append(class_id)
        self._kd_tree = KDTree(self._kd_embeddings, metric="euclidean")
        return class_id

    def _log(self, msg: str):
        self._l.append(msg)

    def get_log(self) -> typing.List[str]:
        return self._l
