from collections import deque
from math import ceil, sqrt

import cv2 # TODO: how should this be handled?
import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin

PLUGIN_NAME = 'beholder'

PARAMETERS = 'parameters'
TENSOR = 'tensor'
NETWORK = 'network'

IMAGE_HEIGHT = 600
IMAGE_WIDTH = IMAGE_HEIGHT * (4.0/3.0)

# TODO: should some of these methods be pulled out into something else?

class Beholder(base_plugin.TBPlugin):

  # TODO: stuff from other plugins: multiplexer, context argument
  def __init__(
      self,
      session,
      logdir,
      variance_duration=5,
      scaling_scope=TENSOR):

    self.LOGDIR = logdir
    self.SCALING_SCOPE = scaling_scope
    self.SESSION = session
    self.VARIANCE_DURATION = variance_duration

    # TODO: store the version with the most computation already done.
    self.arrays_over_time = deque([], variance_duration)


  def _get_mode(self):
    try:
      return pau.RetrieveAsset(self.LOGDIR, PLUGIN_NAME, 'mode')
    except KeyError:
      directory = pau.PluginDirectory(self.LOGDIR, PLUGIN_NAME)
      tf.gfile.MakeDirs(directory)

      with open(directory + '/mode', 'w') as mode_file:
        mode_file.write(PARAMETERS)

      return pau.RetrieveAsset(self.LOGDIR, PLUGIN_NAME, 'mode')


  def _get_arrays(self, mode):

    def get_parameters():
      return [self.SESSION.run(tensor) for tensor in tf.trainable_variables()]

    mode_options = {
        # TODO: add a parameter that allows people to set their own function?
        PARAMETERS: get_parameters,
    }

    return mode_options[mode]()


  def _arrays_to_image(self, arrays):

    global_min = min([np.min(array) for array in arrays])
    global_max = max([np.max(array) for array in arrays])
    column_width = int(IMAGE_WIDTH / len(arrays))

    def reshape_array(array):
      array = np.ravel(array)
      columns = int(ceil(sqrt((column_width * len(array)) / IMAGE_HEIGHT)))
      rows = int(len(array) / columns)

      # Truncate whatever remaining values there are that don't fit. Hopefully,
      # it doesn't matter that the last few (< column count) aren't there.
      return np.reshape(array[:rows * columns], (rows, columns))

    def scale_for_display(array):

      if self.SCALING_SCOPE == TENSOR:
        minimum = np.min(array)
        maximum = np.max(array - minimum)

      elif self.SCALING_SCOPE == NETWORK:
        minimum = global_min
        maximum = global_max

      array -= minimum
      return array * (255 / maximum)

    reshaped_arrays = [reshape_array(array) for array in arrays]
    image_scaled_arrays = [scale_for_display(array)
                           for array in reshaped_arrays]
    final_arrays = [cv2.resize(array,
                               (column_width, IMAGE_HEIGHT),
                               interpolation=cv2.INTER_NEAREST)
                    for array in image_scaled_arrays]

    return np.hstack(final_arrays)


  def update(self):
    mode = self._get_mode()
    arrays = self._get_arrays(mode)
    image = self._arrays_to_image(arrays)
    cv2.imwrite(pau.PluginDirectory(self.LOGDIR, PLUGIN_NAME) + '/frame.png',
                image)


  def get_plugin_apps(self):
    raise NotImplementedError()


  def is_active(self):
    raise NotImplementedError()
