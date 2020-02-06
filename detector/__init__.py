import abc

import typing
import numpy as np


class Detector(object):
    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> (typing.List[typing.List[int]], typing.List[float]):
        raise NotImplementedError
