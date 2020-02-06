import os
import typing
from ml_serving.drivers import driver
import numpy as np

from tools import images
from tools.profiler import Profiler, profiler_pipe


class Facenet(object):
    def __init__(self, facenet_path: str, profiler: Profiler = None):

        ext = os.path.splitext(facenet_path)[1]
        if ext == ".xml":
            self._driver: driver.ServingDriver = driver.load_driver("openvino")()
            self._driver.load_model(facenet_path)
            self._tf = False
        elif ext == ".pb":
            self._driver: driver.ServingDriver = driver.load_driver("tensorflow")()
            self._driver.load_model(
                facenet_path, inputs="input:0", outputs="embeddings:0"
            )
            self._tf = True
        else:
            raise RuntimeError("undefined facenet model")

        self._input_shape = list(self._driver.inputs.values())[0]
        self._input_name = list(self._driver.inputs)[0]
        self._output_name = list(self._driver.outputs)[0]
        self._profiler = profiler_pipe(profiler)

    def embeddings(self, frame: np.ndarray, source: typing.List[typing.List[int]]):
        prf = "face reidentification"
        self._profiler.start(prf)
        face_images = images.get_images(
            frame,
            np.array([s[:4] for s in source]),
            normalization=images.DEFAULT_NORMALIZATION,
        )
        embeddings = []
        for face_img in face_images:
            if self._tf:
                face_img = face_img.reshape([1, 160, 160, 3])
                outputs = self._driver.predict(
                    {"input:0": face_img}
                )
                output = outputs["embeddings:0"][0]
            else:
                face_img = face_img.transpose([2, 0, 1]).reshape(self._input_shape)
                outputs = self._driver.predict({self._input_name: face_img})
                output = outputs[self._output_name]
            embeddings.append(output.reshape([-1]))
        embeddings = (np.asarray(embeddings) + 1.0) / 2
        self._profiler.stop(prf)
        return embeddings
