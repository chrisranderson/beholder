from __future__ import division, print_function

from collections import deque
import json
import time

import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins.beholder import im_util

PLUGIN_NAME = 'beholder'
TAG_NAME = 'beholder-frame'
SUMMARY_FILENAME = 'frame.summary'

SECTION_HEIGHT = 80
IMAGE_WIDTH = 1200

INFO_HEIGHT = 40

DEFAULT_CONFIG = {
    'values': 'trainable_variables',
    'mode': 'variance',
    'scaling': 'layer',
    'window_size': 15,
    'FPS': 10
}


class Beholder():

  def __init__(
      self,
      session,
      logdir):
    '''
      sections_over_time: deque<list<nparray>>. A list of sections is appended
        at every update.
    '''

    self.LOGDIR_ROOT = logdir
    self.PLUGIN_LOGDIR = pau.PluginDirectory(logdir, PLUGIN_NAME)
    self.SESSION = session

    self.sections_over_time = deque([], DEFAULT_CONFIG['window_size'])
    self.frame_placeholder = None
    self.summary_op = None

    self.config = dict(DEFAULT_CONFIG)
    self.old_config = dict(DEFAULT_CONFIG)

    self.last_update_time = time.time()
    self.last_image_height = 0


  @staticmethod
  def gradient_helper(optimizer, loss, var_list=None):
    '''A helper to get the gradients out at each step.

    Args:
      optimizer: the optimizer op.
      loss: the op that computes your loss value.

    Returns: the tensors and the train_step op.
    '''
    if var_list is None:
      var_list = tf.trainable_variables()

    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads = [pair[0] for pair in grads_and_vars]

    return grads, optimizer.apply_gradients(grads_and_vars)


  def _update_config(self):
    '''Reads the config file from disk or creates a new one.'''

    try:
      json_string = pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'config')
      config = json.loads(json_string)
    except (KeyError, ValueError):
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)

      with open(self.PLUGIN_LOGDIR + '/config', 'w') as config_file:
        config_file.write(json.dumps(DEFAULT_CONFIG))

      config = DEFAULT_CONFIG

    for key, value in config.items():
      try:
        config[key] = int(value)
      except ValueError:
        pass

    self.config = config


  def _get_section_info_images(self, arrays, sections):
    '''Renders images that go above each section.

    Args:
      arrays: a list of np arrays.
      sections: unscaled nparrays

    Returns:
      A list of np arrays.
    '''
    images = []

    if self.config['values'] == 'trainable_variables':
      names = [x.name for x in tf.trainable_variables()]
    else:
      names = range(len(arrays))

    for name, array, section in zip(names, arrays, sections):
      minimum = section.min()
      maximum = section.max()
      shape = array.shape
      min_text = 'min: {:.3e}'.format(minimum)
      max_text = 'max: {:.3e}'.format(maximum)
      range_text = 'range: {:.3e}'.format(maximum - minimum)

      template = '{:^30}{:^15}{:^20}{:^20}{:^20}'
      final_text = template.format(name, shape, min_text, max_text, range_text)

      images.append(im_util.text_image(INFO_HEIGHT, IMAGE_WIDTH, final_text))

    return images


  def _sections_to_variance_sections(self):
    '''Computes the variance of corresponding sections over time.

    Returns:
      a list of np arrays.
    '''
    variance_sections = []

    for i in range(len(self.sections_over_time[0])):
      time_sections = [sections[i] for sections in self.sections_over_time]
      variance = np.var(time_sections, axis=0)
      variance_sections.append(variance)

    return variance_sections


  def _arrays_to_image(self, arrays):
    '''
    Args:
      arrays: a list of np arrays to be visualized.

    Returns:
      a numpy array image ready to be turned into a summary.
    '''
    sections = im_util.arrays_to_sections(arrays, SECTION_HEIGHT, IMAGE_WIDTH)
    self.sections_over_time.append(sections)

    if self.config['mode'] == 'variance':
      sections = self._sections_to_variance_sections()

    section_info_images = self._get_section_info_images(arrays, sections)
    scaled_sections = im_util.scale_sections(sections, self.config['scaling'])
    image_stack = []

    for info, section in zip(section_info_images, scaled_sections):
      image_stack.append(info)
      image_stack.append(section)

    return im_util.resize(np.vstack(image_stack).astype(np.uint8),
                          len(arrays) * (SECTION_HEIGHT + INFO_HEIGHT),
                          IMAGE_WIDTH)


  def _write_summary(self, frame):
    '''Writes the frame to disk as a tensor summary.'''

    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    with open(path, 'wb') as file:
      file.write(summary)


  def _maybe_clear_deque(self):
    '''Clears the deque if certain parts of the config have changed.'''
    for config_item in ['values', 'mode']:
      if self.config[config_item] != self.old_config[config_item]:
        self.sections_over_time.clear()
        break

    self.old_config = self.config

    window_size = self.config['window_size']
    if window_size != self.sections_over_time.maxlen:
      self.sections_over_time = deque(self.sections_over_time, window_size)

  def _get_final_image(self, arrays=None, frame=None):


    def enough_time_has_passed():
      '''For limiting how often frames are computed.'''
      if self.config['FPS'] == 0:
        return False
      else:
        earliest_time = self.last_update_time + (1.0 / self.config['FPS'])
        return time.time() >= earliest_time


    final_image = None

    if self.config['values'] == 'frames':
      if frame is None:
        message = "A frame wasn't passed into the update function."
        final_image = im_util.text_image(INFO_HEIGHT, IMAGE_WIDTH, message)
      else:
        frame = frame() if callable(frame) else frame
        final_image = im_util.scale_image_for_display(frame)

    elif enough_time_has_passed():
      # The visualization is a lot smoother if last_update_time is updated here
      # rather than after everything is done.
      self.last_update_time = time.time()
      self._maybe_clear_deque()

      if self.config['values'] == 'trainable_variables':
        arrays = [self.SESSION.run(x) for x in tf.trainable_variables()]
        final_image = self._arrays_to_image(arrays)

      if self.config['values'] == 'arrays':
        if arrays is None:
          message = "Arrays weren't passed into the update function."
          final_image = im_util.text_image(INFO_HEIGHT, IMAGE_WIDTH, message)
        else:
          arrays = arrays if isinstance(arrays, list) else [arrays]
          final_image = self._arrays_to_image(arrays)

    return final_image

  # TODO: blanket try and except for production? I don't someone's script to die
  #       after weeks of running because of a visualization.
  def update(self, arrays=None, frame=None):
    '''Creates a frame and writes it to disk.

    Args:
      arrays: a list of np arrays. Use the "custom" option in the client.
      frame: a 2D np array. This way the plugin can be used for video of any
             kind, not just the visualization that comes with the plugin.

             frame can also be a function, which only is evaluated when the
             "frame" option is selected by the client.

    '''
    self._update_config()
    final_image = self._get_final_image(arrays, frame)

    if final_image is not None:

      image_height, image_width = final_image.shape

      if self.summary_op is None or self.last_image_height != image_height:
        self.frame_placeholder = tf.placeholder(tf.uint8, [image_height,
                                                           image_width])
        self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                    self.frame_placeholder)
      self._write_summary(final_image)
      self.last_image_height = image_height
