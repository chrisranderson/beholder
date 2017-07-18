from collections import deque

import numpy as np
import tensorflow as tf

from tensorboard.plugins.beholder import im_util
from tensorboard.plugins.beholder.shared_config import INFO_HEIGHT,\
  SECTION_HEIGHT, IMAGE_WIDTH, DEFAULT_CONFIG

class Visualizer():

  def __init__(self, session):
    self.sections_over_time = deque([], DEFAULT_CONFIG['window_size'])
    self.config = DEFAULT_CONFIG
    self.old_config = DEFAULT_CONFIG
    self.SESSION = session


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


  def build_frame(self, arrays):
    # The visualization is a lot smoother if last_update_time is updated here
    # rather than after everything is done.
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

  def update(self, config):
    self.config = config
