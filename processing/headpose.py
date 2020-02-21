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

idx_tensor = np.arange(0, 66)
MARGIN_COEF = .4
HEAD_POSE_DRIVER_TYPE = 'openvino'
HEAD_POSE_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
)
HEAD_POSE_THRESHOLDS = '45,35,25'


class HeadPoseFilter(object):
    def __init__(self, **kwargs):
        self._type = 'head pose'
        self._profiler = kwargs.get('profiler', profiler.Profiler())
        head_pose_driver = kwargs.get('head_pose_driver')
        if head_pose_driver is None:
            model_path = kwargs.get('head_pose_model_path') or HEAD_POSE_PATH
            if model_path == 'skip':
                head_pose_driver = None
            else:
                # project https://github.com/kibernetika-ai/deep-head-pose
                head_pose_driver = utils.get_driver(
                    model_path,
                    'head pose',
                    None,
                    model_class='processing.hopenet:HopenetProxy',
                    map_location='cpu'
                )
        self._driver = head_pose_driver
        self._driver_type = head_pose_driver.driver_name
        self._head_pose_axis_threshold = None
        self._head_pose_thresholds = get_thresholds(
            kwargs.get('head_pose_thresholds', HEAD_POSE_THRESHOLDS))
        # self._no_skip = args.get_bool(kwargs, 'head_pose_no_skip')

    def set_thresholds(self, head_pose_thresholds: str):
        self._head_pose_thresholds = get_thresholds(head_pose_thresholds)

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
        if self._driver is None:
            return to_filter
        filtered = []
        poses = self.head_poses(frame, [f.bbox for f in to_filter])
        for i, ind in enumerate(poses):
            f = to_filter[i]
            set_head_pose(f, ind)
            # if not self._no_skip:
            #     if wrong_pose(ind, self._head_pose_thresholds, self._head_pose_axis_threshold):
            #         f.set_skipped(True)
            filtered.append(f)
        return filtered

    def _im_size(self):
        if self._driver_type == HEAD_POSE_DRIVER_TYPE:
            return 60
        if self._driver_type == 'pytorch':
            return 224
        raise ValueError('unknown driver type \'{}\''.format(self._driver_type))

    def head_poses(self, frame, boxes):
        if self._driver is None:
            return []
        if boxes is None or len(boxes) == 0:
            return []

        imgs = np.stack(images.get_images(
            frame, np.array(boxes), self._im_size(), 0, normalization=None, face_crop_margin_coef=MARGIN_COEF))

        return self.head_poses_for_images(imgs, False)

    def head_poses_for_images(self, imgs, resize: bool):

        if self._driver_type == HEAD_POSE_DRIVER_TYPE:
            # Convert to BGR.
            imgs = imgs[:, :, :, ::-1]

            if resize:
                im_size = self._im_size()
                imgs = [cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA) for img in imgs]
            outputs = self._driver.predict({'data': np.array(imgs).transpose([0, 3, 1, 2])})

            yaw = - outputs["angle_y_fc"].reshape([-1])
            pitch = - outputs["angle_p_fc"].reshape([-1])
            roll = outputs["angle_r_fc"].reshape([-1])

            # Return shape [N, 3] as a result
            return np.array([yaw, pitch, roll]).transpose()

        if self._driver_type == 'pytorch':
            ret = []
            if resize:
                im_size = self._im_size()
                imgs = [cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_AREA) for img in imgs]
            imgs = [images.hopenet(img) for img in imgs]

            outputs = self._driver.predict({'0': np.stack(imgs)})
            for i, out in enumerate(outputs['0']):
                yaw = np.sum(special.softmax(outputs['0'][i]) * idx_tensor) * 3 - 99
                pitch = np.sum(special.softmax(outputs['1'][i]) * idx_tensor) * 3 - 99
                roll = np.sum(special.softmax(outputs['2'][i]) * idx_tensor) * 3 - 99
                ret.append([yaw, pitch, roll])

            return np.array(ret)

        raise ValueError('unknown driver type \'{}\''.format(self._driver_type))


def wrong_pose(pose: [float], head_pose_thresholds: [float], head_pose_axis_threshold: float):
    if len(pose) != 3:
        raise ValueError('expected head pose as 3 floats')
    if len(head_pose_thresholds) != 3:
        raise ValueError('expected head pose thresholds as 3 floats')
    # [yaw, pitch, roll]
    [y, p, r] = pose
    if head_pose_axis_threshold is not None:
        _, _, _, z_len = _head_pose_to_axis(pose)
        return z_len > head_pose_axis_threshold
    return (np.abs(y) > head_pose_thresholds[0]
            or np.abs(p) > head_pose_thresholds[1]
            or np.abs(r) > head_pose_thresholds[2])


def set_head_pose(to: tracked_face.TrackedFace, hp_ind: [float]):
    to.metadata['head_pose'] = hp_ind
    axis = _head_pose_to_axis(hp_ind)
    to.metadata['head_pose_axis'] = axis


def _head_pose_to_axis(hp_ind: [float]):
    (yaw, pitch, roll) = hp_ind

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right
    x1 = np.cos(yaw) * np.cos(roll)
    y1 = np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)

    # Y-Axis pointing down
    x2 = -np.cos(yaw) * np.sin(roll)
    y2 = np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)

    # Z-Axis out of the screen
    x3 = np.sin(yaw)
    y3 = -np.cos(yaw) * np.sin(pitch)
    z_len = np.sqrt(x3 ** 2 + y3 ** 2)

    return (x1, y1), (x2, y2), (x3, y3), z_len


def get_thresholds(thresholds: str) -> [int]:
    head_pose_thresholds_spl = thresholds.split(",")
    if len(head_pose_thresholds_spl) != 3:
        raise ValueError('head_pose_thresholds must be three comma separated numbers')
    try:
        ret = [float(f.strip()) for f in head_pose_thresholds_spl]
        return ret
    except Exception:
        raise ValueError('head_pose_thresholds must be three comma separated float numbers')
