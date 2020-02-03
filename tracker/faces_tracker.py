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
    def __init__(self, bbox: [int], prob: float, id: int):
        self.bbox = bbox
        self.prob = prob
        self.id = id

    def label(self) -> str:
        return f'Face {self.id}'


class FacesTracker(object):
    def __init__(self,
                 checkpoint_dir=os.path.join(os.path.dirname(__file__), '..', LOG_DIR, 'checkpoints'),
                 face_detection_path=FACE_DETECTION_PATH,
                 count_add: int = 1, count_remove: int = 5,
                 intersection_threshold: float = .2,
                 gpu_id=GPU_ID,
                 ):

        # initial track id
        self._track_index = 0
        # current active tracks
        self._current_tracks = []
        # tracks candidates to start track (id: face not found count)
        self._candidates_add = {}
        self._count_add = count_add
        # tracks candidates to interrupt (id: face not found count)
        self._candidates_remove = {}
        self._count_remove = count_remove

        # intersection coef for identifying tracked and detected faces
        self._intersection_threshold = intersection_threshold

        self._re3_tracker = re3_tracker.Re3Tracker(checkpoint_dir, gpu_id=gpu_id)

        self._face_detect_driver = driver.load_driver("openvino")()
        self._face_detect_driver.load_model(face_detection_path)

    def track(self, frame: np.ndarray) -> (typing.List[TrackedFace], typing.List[typing.List[float]]):

        bgr_frame = frame[:, :, ::-1]

        # existing tracks
        tracked_faces = self._track(bgr_frame, self._current_tracks)

        # detected faces
        detected_faces = self._detect_faces(bgr_frame)

        result = []

        tracks_has_detected = []

        for detected_face in detected_faces:
            detected_bbox, detected_prob = detected_face[:4], detected_face[4]
            is_tracked = False
            for tracked_i, tracked_face in enumerate(tracked_faces):
                tracked_id = self._current_tracks[tracked_i]
                if tracked_id not in tracks_has_detected:
                    if bbox.box_intersection(detected_bbox, tracked_face) > self._intersection_threshold:
                        is_tracked = True
                        tracks_has_detected.append(tracked_id)
                        self._track_add(bgr_frame, tracked_id, detected_bbox)
                        if self._count_add > 0 and self._track_index in self._candidates_add:
                            if self._candidates_add[self._track_index] > self._count_add:
                                del self._candidates_add[self._track_index]
                                detected = True
                            else:
                                self._candidates_add[self._track_index] += 1
                                detected = False
                        else:
                            detected = True
                        if detected:
                            result.append(TrackedFace(detected_bbox, detected_prob, tracked_id))
                        break
            if not is_tracked:
                self._track_index += 1
                self._current_tracks.append(self._track_index)
                tracks_has_detected.append(self._track_index)
                print(f"!!!! add face ID {self._track_index} with coords {detected_bbox} and prob {detected_prob}")
                self._track_add(bgr_frame, self._track_index, detected_bbox)
                if self._count_add == 0:
                    result.append(TrackedFace(detected_bbox, detected_prob, self._track_index))
                else:
                    self._candidates_add[self._track_index] = 1

        remove_tracks = []
        for i in self._current_tracks:
            if i not in tracks_has_detected:
                if i in self._candidates_add:
                    del self._candidates_add[i]
                    remove_tracks.append(i)
                    continue
                if i not in self._candidates_remove:
                    self._candidates_remove[i] = 0
                self._candidates_remove[i] += 1
                if self._candidates_remove[i] > self._count_remove:
                    del self._candidates_remove[i]
                    remove_tracks.append(i)

        self._current_tracks = [e for e in self._current_tracks if e not in remove_tracks]

        return result, tracked_faces

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
