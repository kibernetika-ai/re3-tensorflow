import glob
import os
import time
import sys
import os.path

from tools.profiler import Profiler, profiler_pipe

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

import cv2
import numpy as np
import tensorflow as tf

from tracker import network
from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.tensorflow_util import tf_util

# Network Constants
from constants import CROP_SIZE
from constants import CROP_PAD
from constants import LSTM_SIZE
from constants import LOG_DIR
from constants import GPU_ID
from constants import MAX_TRACK_LENGTH

SPEED_OUTPUT = True


class Re3Tracker(object):
    def __init__(self, checkpoint_dir=os.path.join(os.path.dirname(__file__), '..', LOG_DIR, 'checkpoints'),
                 gpu_id=GPU_ID, profiler: Profiler = None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        tf.Graph().as_default()
        self._init_tf()
        self.sess = tf_util.Session()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is None:
            raise IOError(
                ('Checkpoint model could not be found. '
                 'Did you download the pretrained weights? '
                 'Download them here: http://bit.ly/2L5deYF and read the Model section of the Readme.'))
        tf_util.restore(self.sess, ckpt.model_checkpoint_path)

        self.tracked_data = {}

        self.time = 0
        self.total_forward_count = -1

        self._profiler: Profiler = profiler_pipe(profiler)

        self._re3_crop_profiler = "re3 cropping image"
        self._re3_crop2_profiler = "re3 cropping image 2"
        self._re3_sess_profiler = "re3 session run"
        self._re3_sess_p1_profiler = "re3 session_p1 run"
        self._re3_sess_p2_profiler = "re3 session_p2 run"
        self._re3_sess2_profiler = "re3 session run 2"
        self._re3_sess_mult_profiler = "re3 session mult run"
        self._re3_sess2_mult_profiler = "re3 session mult run 2"

    def _init_tf(self):
        self.imagePlaceholder1 = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.imagePlaceholder2 = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.fc = tf.placeholder(tf.float32, shape=(None, 1024))
        self.prevLstmState = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        self.fc_out= network.inference_conf(self.imagePlaceholder1,self.imagePlaceholder2, num_unrolls=1,
                                                       batch_size=self.batch_size)
        self.outputs, self.state1, self.state2 = network.inference_single(
            self.fc, num_unrolls=1, batch_size=self.batch_size, train=False,
            prevLstmState=self.prevLstmState)


    def _run_sess(self, croppedInput0, croppedInput1, lstmState, mult=0):
        feed_dict_0 = {
            self.imagePlaceholder1: [croppedInput0],
            self.imagePlaceholder2: [croppedInput1],
            self.batch_size: 1,
        }
        self._profiler.start(self._re3_sess_p1_profiler)
        fc_out = self.sess.run([self.fc_out], feed_dict=feed_dict_0)
        fc_out = fc_out[0]
        self._profiler.stop(self._re3_sess_p1_profiler)
        feed_dict_1 = {
            self.fc: fc_out,
            self.prevLstmState: lstmState,
            self.batch_size: 1,
        }
        self._profiler.start(self._re3_sess_p2_profiler)
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict_1)
        newlstmState = []
        if mult > 0:
            for uu in range(mult):
                newlstmState.append([s1[0][[uu], :], s1[1][[uu], :], s2[0][[uu], :], s2[1][[uu], :]])
        else:
            newlstmState = [s1[0], s1[1], s2[0], s2[1]]
        self._profiler.stop(self._re3_sess_p2_profiler)
        return rawOutput,newlstmState



    # unique_id{str}: A unique id for the object being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_box{None or 4x1 numpy array or list}: 4x1 bounding box in X1, Y1, X2, Y2 format.
    def track(self, unique_id, image, starting_box=None):
        start_time = time.time()

        if type(image) == str:
            image = cv2.imread(image)[:, :, ::-1]
        else:
            image = image

        image_read_time = time.time() - start_time

        if starting_box is not None:
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
            pastBBox = np.array(starting_box)  # turns list into numpy array if not and copies for safety.
            forwardCount = 0
            prev = image
        elif unique_id in self.tracked_data:
            lstmState, pastBBox, prev, originalFeatures, forwardCount = self.tracked_data[unique_id]
        else:
            raise Exception('Unique_id %s with no initial bounding box' % unique_id)
        self._profiler.start(self._re3_crop_profiler)
        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prev, pastBBox, CROP_PAD, CROP_SIZE)
        self._profiler.stop(self._re3_crop_profiler)
        self._profiler.start(self._re3_crop_profiler)
        croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
        self._profiler.stop(self._re3_crop_profiler)

        rawOutput,lstmState = self._run_sess(croppedInput0, croppedInput1, [lstmState])
        #if forwardCount == 0:
        originalFeatures = lstmState

        # Shift output box to full image coordinate system.
        outputBox = bb_util.from_crop_coordinate_system(rawOutput.squeeze() / 10.0, pastBBoxPadded, 1, 1)


        forwardCount += 1
        self.total_forward_count += 1

        if starting_box is not None:
            # Use label if it's given
            outputBox = np.array(starting_box)

        self.tracked_data[unique_id] = (lstmState, outputBox,image, originalFeatures, forwardCount)
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed:   %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed: %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed:      %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        return outputBox

    # unique_ids{list{string}}: A list of unique ids for the objects being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_boxes{None or dictionary of unique_id to 4x1 numpy array or list}: unique_ids to starting box.
    #    Starting boxes only need to be provided if it is a new track. Bounding boxes in X1, Y1, X2, Y2 format.
    def multi_track(self, unique_ids, image, starting_boxes=None):
        outputBoxes = np.zeros((len(unique_ids), 4))
        for i in range(len(unique_ids)):
            b = None
            if starting_boxes is not None:
                b = starting_boxes[unique_ids[i]]
            outputBox = self.track(unique_ids[i],image,starting_box=b)
            outputBoxes[i, :] = outputBox
        return outputBoxes

    def multi_track_0(self, unique_ids, image, starting_boxes=None):
        start_time = time.time()
        assert type(unique_ids) == list, 'unique_ids must be a list for multi_track'
        assert len(unique_ids) > 1, 'unique_ids must be at least 2 elements'

        if type(image) == str:
            image = cv2.imread(image)[:, :, ::-1]
        else:
            image = image.copy()

        image_read_time = time.time() - start_time

        # Get inputs for each track.
        images_prev = []
        images_cur = []
        lstmStates = [[] for _ in range(4)]
        pastBBoxesPadded = []
        if starting_boxes is None:
            starting_boxes = dict()
        for unique_id in unique_ids:
            if unique_id in starting_boxes:
                lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                pastBBox = np.array(
                    starting_boxes[unique_id])  # turns list into numpy array if not and copies for safety.
                prevImage = image
                originalFeatures = None
                forwardCount = 0
                self.tracked_data[unique_id] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
            elif unique_id in self.tracked_data:
                lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            else:
                raise Exception('Unique_id %s with no initial bounding box' % unique_id)

            self._profiler.start(self._re3_crop_profiler)
            croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
            croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            self._profiler.stop(self._re3_crop_profiler)
            pastBBoxesPadded.append(pastBBoxPadded)
            images_prev.append(croppedInput0)
            images_cur.append(croppedInput1)
            # images.extend([croppedInput0, croppedInput1])
            for ss, state in enumerate(lstmState):
                lstmStates[ss].append(state.squeeze())

        lstmStateArrays = []
        for state in lstmStates:
            lstmStateArrays.append(np.array(state))

        rawOutput, newStates = self._run_sess(images_prev, images_cur, lstmStateArrays,mult=len(unique_ids))
        outputBoxes = np.zeros((len(unique_ids), 4))
        for uu, unique_id in enumerate(unique_ids):
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            lstmState = newStates[uu]
            if forwardCount == 0:
                originalFeatures = lstmState

            # Shift output box to full image coordinate system.
            pastBBoxPadded = pastBBoxesPadded[uu]
            outputBox = bb_util.from_crop_coordinate_system(rawOutput[uu, :].squeeze() / 10.0, pastBBoxPadded, 1, 1)

            forwardCount += 1
            self.total_forward_count += 1

            if unique_id in starting_boxes:
                # Use label if it's given
                outputBox = np.array(starting_boxes[unique_id])

            outputBoxes[uu, :] = outputBox
            self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed per object: %.3f FPS' % (
                    len(unique_ids) / (end_time - start_time - image_read_time)))
            print('Current tracking speed per frame:  %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed:          %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed per object:    %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        return outputBoxes


class CopiedRe3Tracker(Re3Tracker):
    def __init__(self, sess, copy_vars, gpu=None):
        self.sess = sess
        self.imagePlaceholder = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.prevLstmState = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        network_scope = 'test_network'
        if gpu is not None:
            with tf.device('/gpu:' + str(gpu)):
                with tf.variable_scope(network_scope):
                    self.outputs, self.state1, self.state2 = network.inference(
                        self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                        prevLstmState=self.prevLstmState)
        else:
            with tf.variable_scope(network_scope):
                self.outputs, self.state1, self.state2 = network.inference(
                    self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                    prevLstmState=self.prevLstmState)
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_scope)
        self.sync_op = self.sync_from(copy_vars, local_vars)

        self.tracked_data = {}

        self.time = 0
        self.total_forward_count = -1

    def reset(self):
        self.tracked_data = {}
        self.sess.run(self.sync_op)

    def sync_from(self, src_vars, dst_vars):
        sync_ops = []
        with tf.name_scope('Sync'):
            for (src_var, dst_var) in zip(src_vars, dst_vars):
                sync_op = tf.assign(dst_var, src_var)
                sync_ops.append(sync_op)
        return tf.group(*sync_ops)
