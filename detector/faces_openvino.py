import cv2
import typing
from ml_serving.drivers import driver
import numpy as np

from detector import Detector
from tools import bbox

FACE_DETECTION_PATH = (
    "/opt/intel/openvino/deployment_tools/intel_models/"
    "face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
)


class FacesOpenvino(Detector):
    def __init__(
        self,
        face_detection_path: str = FACE_DETECTION_PATH,
        threshold: float = 0.5,
        split_counts: str = "2",
        face_min_size: int = 10,
    ):
        self._driver: driver.ServingDriver = driver.load_driver("openvino")()
        self._driver.load_model(face_detection_path)
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
