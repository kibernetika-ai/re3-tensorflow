import cv2
import typing
from ml_serving.drivers import driver
import numpy as np

from detector import detector

FACE_DETECTION_PATH = (
    "/opt/intel/openvino/deployment_tools/intel_models/"
    "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
)


class FacesOpenvino(detector.Detector):
    def __init__(
        self,
        face_detection_path: str = FACE_DETECTION_PATH,
        threshold: float = 0.5,
        split_counts: str = "2",
        face_min_size: int = 10,
    ):
        super().__init__(threshold, split_counts, face_min_size)
        self._driver: driver.ServingDriver = driver.load_driver("openvino")()
        self._driver.load_model(face_detection_path)

    def _detect_faces_split(self, bgr_frame: np.ndarray, offset=(0, 0)):

        # Get boxes shaped [N, 5]:
        # xmin, ymin, xmax, ymax, confidence
        input_name, input_shape = list(self._driver.inputs.items())[0]
        output_name = list(self._driver.outputs)[0]
        # TODO use images instead
        inference_frame = cv2.resize(
            bgr_frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA
        )
        inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
        outputs = self._driver.predict({input_name: inference_frame})
        output = outputs[output_name]
        output = output.reshape(-1, 7)
        bboxes_raw = output[output[:, 2] > self._threshold]
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
            boxes = boxes[
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                >= self._face_min_area
            ]
        return boxes
