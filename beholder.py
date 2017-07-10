from collections import deque
import json

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
import tensorboard.plugins.beholder.image_util as im_util


CUSTOM = 'custom'
PARAMETERS = 'parameters'

CURRENT = 'current'
VARIANCE = 'variance'

SCALE_LAYER = 'layer'
SCALE_NETWORK = 'network'

PLUGIN_NAME = 'beholder'
TAG_NAME = 'beholder-frame'
SUMMARY_FILENAME = 'frame.summary'

IMAGE_HEIGHT = 600
IMAGE_WIDTH = int(IMAGE_HEIGHT * (4.0/3.0))

DEFAULT_CONFIG = {
    'values': PARAMETERS,
    'mode': VARIANCE,
    'scaling': SCALE_LAYER
}


class Beholder():

  def __init__(
      self,
      session,
      logdir,
      variance_duration=5):

    self.LOGDIR_ROOT = logdir
    self.PLUGIN_LOGDIR = pau.PluginDirectory(logdir, PLUGIN_NAME)
    self.SESSION = session
    self.VARIANCE_DURATION = variance_duration

    self.frames_over_time = deque([], variance_duration)
    self.frame_placeholder = None
    self.summary_op = None
    self.old_values = None


  @staticmethod
  def gradient_helper(optimizer, loss, var_list=None):
    '''
    A helper to get the gradients out at each step.

    Returns: the tensors and the train_step op
    '''
    if var_list is None:
      var_list = tf.trainable_variables()

    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads = [pair[0] for pair in grads_and_vars]

    return grads, optimizer.apply_gradients(grads_and_vars)



  def _get_config(self):
    try:
      json_string = pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'config')
      print('json_string', json_string)
      return json.loads(json_string)
    except (KeyError, ValueError):
      print('Could not read config file. Creating a config file.')
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)

      with open(self.PLUGIN_LOGDIR + '/config', 'w') as config_file:
        config_file.write(json.dumps(DEFAULT_CONFIG))

      return DEFAULT_CONFIG


  def _get_display_frame(self, config, arrays=None):
    values, mode, scaling = config['values'], config['mode'], config['scaling']

    if values != self.old_values:
      self.frames_over_time.clear()

    self.old_values = values

    if values != CUSTOM:
      arrays = [self.SESSION.run(x) for x in tf.trainable_variables()]

    global_min, global_max = im_util.global_extrema(arrays)
    absolute_frame = im_util.arrays_to_image(arrays, scaling,
                                             IMAGE_HEIGHT, IMAGE_WIDTH)

    self.frames_over_time.append(absolute_frame)

    if mode == CURRENT:
      final_frame = absolute_frame

    elif mode == VARIANCE:
      stacked = np.dstack(self.frames_over_time)
      variance = np.var(stacked, axis=2)
      final_frame = im_util.scale_for_display(variance, scaling,
                                              global_min, global_max)

    return final_frame


  def _write_summary(self, frame):
    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })

    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    with open(path, 'wb') as file:
      file.write(summary)


  def update(self, arrays=None, frame=None):
    config = self._get_config()

    print('config', config)

    if frame is None:
      frame = self._get_display_frame(config, arrays)

    if self.summary_op is not None:
      self._write_summary(frame)
    else:
      self.frame_placeholder = tf.placeholder(tf.float32,
                                              [IMAGE_HEIGHT, IMAGE_WIDTH])
      self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                  self.frame_placeholder)
      self._write_summary(frame)
