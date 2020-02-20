import logging
import time
import typing

import cv2
import numpy as np

from tracker import tracked_face
from tools import images
from tools import profiler
import utils.special as special
from utils import utils

LOG = logging.getLogger(__name__)

idx_tensor = np.arange(0, 101)
MARGIN_COEF = .4
AGE_GENDER_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'
)
AGE_GENDER_DRIVER_TYPE = 'openvino'


class AgeGenderFilter(object):
    def __init__(self, **kwargs):
        # Do not filter, just save filter data.
        self._type = 'age gender'
        self._profiler = kwargs.get('profiler', profiler.Profiler())
        self.filter_keep = True
        model_path = kwargs.get('age_model_path')
        self.single_driver = not bool(model_path)
        if self.single_driver:
            age_gender_driver = kwargs.get('age_gender_driver')
            self._driver_type = kwargs.get('age_gender_driver_type', AGE_GENDER_DRIVER_TYPE)
            if age_gender_driver is None:
                model_path = kwargs.get('age_gender_model_path', AGE_GENDER_PATH)
                if self._driver_type == AGE_GENDER_DRIVER_TYPE:
                    age_gender_driver = utils.get_driver(model_path, 'age gender')
                else:
                    raise ValueError(f"unknown {self._type} driver type '{self._driver_type}'")
            self._driver = age_gender_driver
        else:
            age_driver = kwargs.get('age_driver')
            gender_driver = kwargs.get('gender_driver')
            if gender_driver is None:
                # Load default
                model_path = kwargs.get('age_gender_model_path', AGE_GENDER_PATH)
                gender_driver = utils.get_driver(model_path, 'age gender')
                self._gender_driver_type = gender_driver.driver_name

            if age_driver is None:
                model_path = kwargs.get('age_model_path')
                age_driver = utils.get_driver(model_path, 'age')
                self._age_driver_type = age_driver.driver_name

            self._age_driver = age_driver
            self._gender_driver = gender_driver

    def filter(self,
               frame: np.ndarray,
               faces: typing.List[tracked_face.TrackedFace],
               ) -> typing.List[tracked_face.TrackedFace]:
        start = time.time()
        # ret = [f for f in faces if f.is_skipped()] if self.filter_keep else []
        ret = []
        to_filter = faces
        filtered = self._filter(frame, to_filter)
        # if not self.filter_keep:
        #     filtered = [f for f in filtered if not f.is_skipped()]
        ret.extend(filtered)
        self._profiler.add('{} filter, sec'.format(self._type), time.time() - start)
        return ret

    def _filter(self, frame: np.ndarray,
                to_filter: typing.List[tracked_face.TrackedFace]) -> typing.List[tracked_face.TrackedFace]:
        filtered = []
        age_genders = self.age_gender(frame, [f.bbox for f in to_filter])
        for i, age_gender_data in enumerate(age_genders):
            f = to_filter[i]
            set_age_gender(f, age_gender_data)
            filtered.append(f)
        return filtered

    def _im_size(self):
        if self.single_driver:
            if self._driver_type == AGE_GENDER_DRIVER_TYPE:
                return 62
        else:
            if self._age_driver_type == AGE_GENDER_DRIVER_TYPE:
                age_size = 62
            else:
                age_size = 224

            gender_size = 62
            if self._gender_driver_type == AGE_GENDER_DRIVER_TYPE:
                gender_size = 62

            return age_size, gender_size

        raise ValueError(f"unknown {self._type} driver type '{self._driver_type}'")

    def age_gender_without_boxes(self, img):
        if self.single_driver:
            if self._driver is None:
                return []
            im_size = self._im_size(), self._im_size()
            imgs = np.stack(
                [cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)]
            )

            return self.age_gender_for_images(imgs)
        else:
            if self._age_driver is None or self._gender_driver is None:
                return []

            age_size, gender_size = self._im_size()
            age_imgs = np.stack(
                [cv2.resize(img, (age_size, age_size), interpolation=cv2.INTER_CUBIC)]
            )
            gender_imgs = np.stack(
                [cv2.resize(img, (gender_size, gender_size), interpolation=cv2.INTER_CUBIC)]
            )
            return self.age_gender_for_images(age_imgs, gender_imgs)

    def age_gender(self, frame, boxes):
        if boxes is None or len(boxes) == 0:
            return []

        if self.single_driver:
            if self._driver is None:
                return []
            imgs = np.stack(images.get_images(
                frame, np.array(boxes),
                self._im_size(),
                face_crop_margin=0,
                normalization=None,
                face_crop_margin_coef=MARGIN_COEF
            ))

            return self.age_gender_for_images(imgs)
        else:
            if self._age_driver is None or self._gender_driver is None:
                return []

            age_size, gender_size = self._im_size()
            age_imgs = np.stack(images.get_images(
                frame, np.array(boxes),
                age_size,
                face_crop_margin=0,
                normalization=None,
                face_crop_margin_coef=MARGIN_COEF
            ))
            gender_imgs = np.stack(images.get_images(
                frame, np.array(boxes),
                gender_size,
                face_crop_margin=0,
                normalization=None,
                face_crop_margin_coef=MARGIN_COEF
            ))
            return self.age_gender_for_images(age_imgs, gender_imgs)

    def age_gender_for_images(self, age_imgs, gender_imgs=None):
        if self.single_driver:
            if self._driver_type == AGE_GENDER_DRIVER_TYPE:
                return self._age_gender_openvino(age_imgs, self._driver)
        else:
            if self._age_driver_type == 'openvino':
                age_gender = self._age_gender_openvino(age_imgs, self._age_driver)
                ages = age_gender[:, 0]
            elif self._age_driver_type == 'tensorflow':
                prepared = age_imgs.transpose([0, 3, 1, 2])
                # prepared = np.expand_dims(prepared, axis=0)
                outputs = []
                for img in prepared:
                    infer_out = self._age_driver.predict({'input.1': np.expand_dims(img, axis=0).astype(np.float)})
                    output = special.softmax(list(infer_out.values())[0])
                    outputs.append(output[0])

                ages = (np.stack(outputs) * idx_tensor).sum(axis=-1)
            else:
                ages = None

            if self._gender_driver_type == 'openvino':
                age_gender = self._age_gender_openvino(gender_imgs, self._gender_driver)
                genders = age_gender[:, 1]
            else:
                genders = None

            return np.stack([ages, genders]).transpose()

        raise ValueError(f"unknown {self._type} driver type '{self._driver_type}'")

    @staticmethod
    def _age_gender_openvino(imgs, age_gender_driver):
        # Convert to BGR.
        imgs = imgs[:, :, :, ::-1]
        outputs = age_gender_driver.predict({'data': np.array(imgs).transpose([0, 3, 1, 2])})

        age = (outputs["age_conv3"].reshape([-1]) * 100).round().astype(int)
        # gender: 0 - female, 1 - male
        gender = outputs['prob'].reshape([-1, 2]).argmax(1)

        # Return shape [N, 2] as a result
        return np.array([age, gender]).transpose()


def set_age_gender(to: tracked_face.TrackedFace, age_gender_data: tuple):
    to.metadata['age'] = round(age_gender_data[0])
    to.metadata['gender'] = age_gender_data[1]
