import os

from detector import Detector
from detector import faces_openvino, faces_tfopencv


def get_face_detector(face_detector_path) -> Detector:
    if face_detector_path is None:
        if os.path.isfile(faces_openvino.FACE_DETECTION_PATH):
            face_detector_path = faces_openvino.FACE_DETECTION_PATH
            print(
                f"detected default openvino face detection model {face_detector_path}, using it"
            )
    if (
        os.path.isfile(face_detector_path)
        and os.path.splitext(face_detector_path)[1] == ".xml"
    ):
        return faces_openvino.FacesOpenvino(face_detector_path)
    if os.path.isdir(face_detector_path) and os.path.isfile(
        os.path.join(face_detector_path, "opencv_face_detector_uint8.pb")
    ):
        return faces_tfopencv.FacesTFOpenCV(face_detector_path)
    raise RuntimeError(f"unable to identify detector for {face_detector_path}")
