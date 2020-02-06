import os

import cv2
from ml_serving.drivers import driver
import numpy as np

from detector import Detector


class FacesTFOpenCV(Detector):
    def __init__(
        self,
        model_path,
        threshold: float = 0.15,
        split_counts: str = "2",
        face_min_size: int = 10,
        use_tensor_rt: bool = False,
    ):
        super().__init__(threshold, split_counts, face_min_size)
        self._model_path = model_path
        _model = "opencv_face_detector_uint8.pb"
        self._driver: driver.ServingDriver = driver.load_driver("tensorflow")()
        self._driver.load_model(
            os.path.join(self._model_path, _model),
            inputs="data:0",
            outputs="mbox_loc:0,mbox_conf_flatten:0",
        )

        if use_tensor_rt:
            _model = "opencv_face_detector_uint8_rt_fp16.pb"

        configFile = self._model_path + "/detector.pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(None, configFile)

        self.prior = np.fromfile(self._model_path + "/mbox_priorbox.np", np.float32)
        self.prior = np.reshape(self.prior, (1, 2, 35568))
        # Dry run
        self.detect(np.zeros((300, 300, 3), np.uint8))

    def _detect_faces_split(self, bgr_frame: np.ndarray, offset=(0, 0)):
        blob = cv2.dnn.blobFromImage(
            bgr_frame[:, :, ::-1], 1.0, (300, 300), [104, 117, 123], False, False
        )
        blob = np.transpose(blob, (0, 2, 3, 1))
        result = self._driver.predict({"data:0": blob})
        probs = result.get("mbox_conf_flatten:0")
        boxes = result.get("mbox_loc:0")
        self.net.setInput(boxes, name="mbox_loc")
        self.net.setInput(probs, name="mbox_conf_flatten")
        self.net.setInput(self.prior, name="mbox_priorbox")
        detections = self.net.forward()

        output = detections.reshape(-1, 7)
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

        return boxes
