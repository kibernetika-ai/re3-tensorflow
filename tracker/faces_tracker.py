import os

import cv2
import typing
from ml_serving.drivers import driver
import numpy as np

from constants import GPU_ID
from constants import LOG_DIR
from tools import bbox
from tracker import re3_tracker

FACE_DETECTION_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
)


class TrackedFace(object):

    _id_counter = 0
    _confirm_after = 1
    _remove_after = 5

    def __init__(self, bbox: [int], prob: float):
        TrackedFace._id_counter += 1
        self._id: int = self._id_counter
        self._bbox: typing.List[int] = bbox
        self._prob: float = prob
        self._just_detected: bool = True
        self._confirmed: bool = self._confirm_after == 0
        self._confirm_count: int = 0 if self._confirmed else 1
        self._removed: bool = False
        self._remove_count: int = 0

    def update(self, bbox: [int], detected: bool = False, prob: float = None):
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

    @property
    def id(self) -> int:
        return self._id

    @property
    def bbox(self) -> typing.List[int]:
        return self._bbox

    @property
    def removed(self) -> bool:
        return self._removed

    @property
    def just_detected(self) -> bool:
        return self._just_detected

    @property
    def remove_count(self) -> int:
        return self._remove_count

    @property
    def confirm_count(self) -> int:
        return self._confirm_count

    @property
    def confirmed(self) -> bool:
        return self._confirmed


class FacesTracker(object):
    def __init__(self,
                 checkpoint_dir=os.path.join(os.path.dirname(__file__), '..', LOG_DIR, 'checkpoints'),
                 face_detection_path=FACE_DETECTION_PATH,
                 intersection_threshold: float = .2,
                 detect_each: int = 10,
                 gpu_id=GPU_ID,
                 ):

        # intersection coef for identifying tracked and detected faces
        self._intersection_threshold: float = intersection_threshold

        self._re3_tracker: re3_tracker.Re3Tracker = re3_tracker.Re3Tracker(checkpoint_dir, gpu_id=gpu_id)

        self._face_detect_driver: driver.ServingDriver = driver.load_driver("openvino")()
        self._face_detect_driver.load_model(face_detection_path)

        self._detect_each: int = detect_each
        self._counter: int = -1

        self._tracked: typing.List[TrackedFace] = []

    def track(self, frame: np.ndarray) -> typing.List[TrackedFace]:

        bgr_frame = frame[:, :, ::-1]

        self._counter += 1

        # existing tracks
        tracked_faces = self._track(bgr_frame, [t.id for t in self._tracked])

        if self._counter % self._detect_each > 0:
            for i, t in enumerate(tracked_faces):
                self._tracked[i].update(t)
            return self._tracked

        # detected faces
        detected_faces = self._detect_faces(bgr_frame)
        detected_track_ids = []

        for detected_face in detected_faces:

            detected_bbox, detected_prob = detected_face[:4], detected_face[4]
            is_tracked = False
            track = None

            for t in self._tracked:
                if t.id not in detected_track_ids:
                    if bbox.box_intersection(detected_bbox, t.bbox) > self._intersection_threshold:
                        is_tracked = True
                        t.update(detected_bbox, detected=True, prob=detected_prob)
                        track = t
                        break

            if not is_tracked:
                track = TrackedFace(detected_bbox, detected_prob)
                self._tracked.append(track)

            if track is not None:
                detected_track_ids.append(track.id)
                self._track_add(bgr_frame, track.id, detected_bbox)

        for tr in self._tracked:
            if tr.id not in detected_track_ids:
                tr.set_not_detected()

        self._tracked = [t for t in self._tracked if not t.removed]

        return self._tracked

    def _track(self, bgr_frame: np.ndarray, indexes: [int]):
        if len(indexes) == 0:
            return []
        if len(indexes) == 1:
            return [self._re3_tracker.track(f'{indexes[0]}', bgr_frame)]
        return self._re3_tracker.multi_track([f'{i}' for i in indexes], bgr_frame)

    def _track_add(self, bgr_frame: np.ndarray, index: int, bbox: [int]):
        self._re3_tracker.track(f'{index}', bgr_frame, bbox)

    def _detect_faces(self, bgr_frame: np.ndarray,
                                threshold: float = 0.5, offset=(0, 0)):
        drv = self._face_detect_driver
        # Get boxes shaped [N, 5]:
        # xmin, ymin, xmax, ymax, confidence
        input_name, input_shape = list(drv.inputs.items())[0]
        output_name = list(drv.outputs)[0]
        inference_frame = cv2.resize(bgr_frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
        outputs = drv.predict({input_name: inference_frame})
        output = outputs[output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > threshold]
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
        return boxes
