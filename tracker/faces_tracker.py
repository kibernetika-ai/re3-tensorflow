import os

import typing
from ml_serving.drivers import driver
import numpy as np

from constants import GPU_ID
from constants import LOG_DIR
from detector import Detector
from detector.getter import get_face_detector
from tools import bbox
from tools import images
from tools.profiler import Profiler, profiler_pipe
from tracker import facenet
from tracker import re3_tracker
from tracker.tracked_face import TrackedFace

from sklearn.neighbors import KDTree


class FacesTracker(object):
    def __init__(
        self,
        re3_checkpoint_dir: str = os.path.join(
            os.path.dirname(__file__), "..", LOG_DIR, "checkpoints"
        ),
        face_detection_path: str = None,
        facenet_path: str = None,
        intersection_threshold: float = 0.2,
        detect_each: int = 10,
        gpu_id=GPU_ID,
        add_min=0.3,
        add_max=0.5,
        profiler: Profiler = None,
    ):

        self._profiler: Profiler = profiler_pipe(profiler)

        # intersection coef for identifying tracked and detected faces
        self._intersection_threshold: float = intersection_threshold

        self._re3_tracker: re3_tracker.Re3Tracker = re3_tracker.Re3Tracker(
            re3_checkpoint_dir, gpu_id=gpu_id, profiler=self._profiler
        )

        self._face_detector: Detector = get_face_detector(face_detection_path)

        if facenet_path is None:
            self._facenet = None
        else:
            self._facenet = facenet.Facenet(facenet_path, self._profiler)

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

        # detected faces: bounding boxes and probabilities
        prf = "face detection"
        self._profiler.start(prf)
        detected_bboxes, detected_probs = self._face_detector.detect(bgr_frame)
        self._profiler.stop(prf)
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
                self._track_add(bgr_frame, track.id, detected_bbox, is_tracked)

        # mark existing tracks as de
        for tr in self._tracked:
            if tr.id not in detected_track_ids:
                tr.set_not_detected()

        # clear removed tracks
        self._tracked = [t for t in self._tracked if not t.removed]

        if self._facenet is None:
            return self._tracked

        # classify tracked faces
        classified_tracks = [
            t for t in self._tracked if t.confirmed and not t.to_remove
        ]
        if len(classified_tracks) > 0:
            confirmed_tracks_bboxes = [t.bbox for t in classified_tracks]
            embs = self._facenet.embeddings(frame, confirmed_tracks_bboxes)
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
        self._profiler.add('re3 tracks', len(indexes))
        prf = "re3 tracking"
        self._profiler.start(prf)
        if len(indexes) == 1:
            tracks = [self._re3_tracker.track(f"{indexes[0]}", bgr_frame)]
        else:
            tracks = self._re3_tracker.multi_track([f"{i}" for i in indexes], bgr_frame)
        self._profiler.stop(prf)
        return tracks

    def _track_add(self, bgr_frame: np.ndarray, index: int, bbox: [int], is_tracked: bool):
        self._profiler.add('re3 tracks {}'.format('updated' if is_tracked else 'added'), 1)
        prf = "re3 track {}".format('updating' if is_tracked else 'adding')
        self._profiler.start(prf)
        self._re3_tracker.track(f"{index}", bgr_frame, bbox)
        self._profiler.stop(prf)

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

    @property
    def log(self) -> typing.List[str]:
        return self._l

    @property
    def profiler(self) -> Profiler:
        return self._profiler
