from collections import deque
import json
import time

import cv2
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
    'scaling': SCALE_LAYER,
    'window_size': 15,
    'FPS': 30
}


class Beholder():

  def __init__(
      self,
      session,
      logdir):

    self.LOGDIR_ROOT = logdir
    self.PLUGIN_LOGDIR = pau.PluginDirectory(logdir, PLUGIN_NAME)
    self.SESSION = session

    self.frames_over_time = deque([], DEFAULT_CONFIG['window_size'])
    self.frame_placeholder = None
    self.summary_op = None
    self.old_config = dict(DEFAULT_CONFIG)
    self.FPS = DEFAULT_CONFIG['FPS']
    self.last_update_time = time.time()


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
      config = json.loads(json_string)
    except (KeyError, ValueError):
      print('Could not read config file. Creating a config file.')
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)

      with open(self.PLUGIN_LOGDIR + '/config', 'w') as config_file:
        config_file.write(json.dumps(DEFAULT_CONFIG))

      config = DEFAULT_CONFIG

    for key, value in config.items():
      try:
        config[key] = int(value)
      except ValueError:
        pass

    return config


  def _get_display_frame(self, config, arrays=None):
    values, mode, scaling = config['values'], config['mode'], config['scaling']

    if values != CUSTOM:
      arrays = [self.SESSION.run(x) for x in tf.trainable_variables()]

    if values == CUSTOM and not arrays:
      return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

    columns = im_util.arrays_to_columns(arrays, IMAGE_HEIGHT, IMAGE_WIDTH)
    self.frames_over_time.append(columns)

    if mode == CURRENT:
      scaled_columns = im_util.scale_for_display(columns, scaling)
    elif mode == VARIANCE:
      variance_columns = []

      for i in range(len(columns)):
        variance = np.var([columns[i] for columns in self.frames_over_time],
                          axis=0)
        variance_columns.append(variance)

      scaled_columns = im_util.scale_for_display(variance_columns, scaling)

    return cv2.resize(np.hstack(scaled_columns).astype(np.uint8),
                      (IMAGE_WIDTH, IMAGE_HEIGHT),
                      interpolation=cv2.INTER_NEAREST)


  def _write_summary(self, frame):
    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    with open(path, 'wb') as file:
      file.write(summary)


  def _update_deque(self, config):

    if config['values'] != self.old_config['values'] or\
       config['mode'] != self.old_config['mode'] or\
       config['scaling'] != self.old_config['scaling']:
      self.frames_over_time.clear()

    self.old_config = config

    window_size = config['window_size']

    if window_size != self.frames_over_time.maxlen:
      self.frames_over_time = deque(self.frames_over_time, window_size)


  def _enough_time_has_passed(self, FPS):
    if FPS == 0:
      return False
    else:
      earliest_time = self.last_update_time + (1.0 / FPS)
      return time.time() >= earliest_time


  def update(self, arrays=None, frame=None):
    config = self._get_config()

    if self._enough_time_has_passed(config['FPS']):
      self._update_deque(config)

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

      self.last_update_time = time.time()
