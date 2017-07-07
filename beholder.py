from collections import deque

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
import tensorboard.plugins.beholder.image_util as im_util


PARAMETER_VARIANCE = 'parameter_variance'
PARAMETERS = 'parameters'

SCALE_LAYER = 'layer'
SCALE_NETWORK = 'network'

PLUGIN_NAME = 'beholder'
TAG_NAME = 'beholder-frame'
SUMMARY_FILENAME = 'frame.summary'

IMAGE_HEIGHT = 600
IMAGE_WIDTH = int(IMAGE_HEIGHT * (4.0/3.0))


class Beholder():

  def __init__(
      self,
      session,
      logdir,
      variance_duration=5,
      scaling_scope=SCALE_LAYER):

    self.LOGDIR_ROOT = logdir
    self.PLUGIN_LOGDIR = pau.PluginDirectory(logdir, PLUGIN_NAME)
    self.SCALING_SCOPE = scaling_scope
    self.SESSION = session
    self.VARIANCE_DURATION = variance_duration

    self.frames_over_time = deque([], variance_duration)
    self.frame_placeholder = None
    self.summary_op = None
    self.variables_op = tf.trainable_variables()


  def _get_mode(self):
    try:
      return pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'mode')
    except KeyError:
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)

      with open(self.PLUGIN_LOGDIR + '/mode', 'w') as mode_file:
        mode_file.write(PARAMETER_VARIANCE)

      return pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'mode')


  def _get_display_frame(self, mode):
    arrays = [self.SESSION.run(x) for x in self.variables_op]
    global_min, global_max = im_util.global_extrema(arrays)
    absolute_frame = im_util.arrays_to_image(arrays, self.SCALING_SCOPE,
                                             IMAGE_HEIGHT, IMAGE_WIDTH)

    self.frames_over_time.append(absolute_frame)

    def get_parameters():
      return absolute_frame

    def get_parameter_variance():
      stacked = np.dstack(self.frames_over_time)
      variance = np.var(stacked, axis=2)
      scaled_frame = im_util.scale_for_display(variance, self.SCALING_SCOPE,
                                               global_min, global_max)
      return scaled_frame


    mode_options = {
        # TODO: add a parameter that allows people to set their own function?
        PARAMETERS: get_parameters,
        PARAMETER_VARIANCE: get_parameter_variance,
    }

    return mode_options[mode]()


  def _write_summary(self, frame):
    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })

    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    with open(path, 'wb') as file:
      file.write(summary)


  def update(self, frame=None):
    mode = self._get_mode()

    if frame is None:
      frame = self._get_display_frame(mode)

    if self.summary_op is not None:
      self._write_summary(frame)
    else:
      self.frame_placeholder = tf.placeholder(tf.float32,
                                              [IMAGE_HEIGHT, IMAGE_WIDTH])
      self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                  self.frame_placeholder)
      self._write_summary(frame)
