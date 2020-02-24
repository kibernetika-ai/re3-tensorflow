import tensorflow as tf
from tracker import network
from re3_utils.tensorflow_util import tf_util

# Network Constants
from constants import CROP_SIZE

from tensorflow import graph_util


class Re3Tracker(object):
    def __init__(self, checkpoint_dir):
        tf.Graph().as_default()
        self._init_tf()
        self.sess = tf_util.Session()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        tf_util.restore(self.sess, ckpt.model_checkpoint_path)

    def _init_tf(self):
        self.imagePlaceholder1 = tf.placeholder(tf.uint8, shape=(1, CROP_SIZE, CROP_SIZE, 3))
        self.imagePlaceholder2 = tf.placeholder(tf.uint8, shape=(1, CROP_SIZE, CROP_SIZE, 3))
        self.fc_out = network.inference_conf(self.imagePlaceholder1, self.imagePlaceholder2, num_unrolls=1,
                                             batch_size=1)
        print(self.fc_out)

    def convert(self):
        output_graph_def = graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(),
                                                                     ['re3/fc6/Relu'])
        f = tf.gfile.GFile('./models/re3/re3.pb', "wb")
        f.write(output_graph_def.SerializeToString())
        f.close()

if __name__ == "__main__":
    r = Re3Tracker('./models/re3')
    r.convert()


