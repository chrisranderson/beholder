from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
from math import floor, sqrt
import pickle

import numpy as np
import tensorflow as tf

from beholder import im_util
from beholder.shared_config import SECTION_HEIGHT, IMAGE_WIDTH, DEFAULT_CONFIG,\
  SECTION_INFO_FILENAME
from beholder.file_system_tools import write_pickle

MIN_SQUARE_SIZE = 4

class Visualizer():

  def __init__(self, session, logdir):
    self.logdir = logdir
    self.sections_over_time = deque([], DEFAULT_CONFIG['window_size'])
    self.config = DEFAULT_CONFIG
    self.old_config = DEFAULT_CONFIG
    self.SESSION = session


  def _conv_section(self, array, section_height, image_width):
    '''Reshape a rank 4 array to be rank 2, where each column of block_width is
    a filter, and each row of block height is an input channel. For example:

    [[[[ 11,  21,  31,  41],
       [ 51,  61,  71,  81],
       [ 91, 101, 111, 121]],
      [[ 12,  22,  32,  42],
       [ 52,  62,  72,  82],
       [ 92, 102, 112, 122]],
      [[ 13,  23,  33,  43],
       [ 53,  63,  73,  83],
       [ 93, 103, 113, 123]]],
     [[[ 14,  24,  34,  44],
       [ 54,  64,  74,  84],
       [ 94, 104, 114, 124]],
      [[ 15,  25,  35,  45],
       [ 55,  65,  75,  85],
       [ 95, 105, 115, 125]],
      [[ 16,  26,  36,  46],
       [ 56,  66,  76,  86],
       [ 96, 106, 116, 126]]],
     [[[ 17,  27,  37,  47],
       [ 57,  67,  77,  87],
       [ 97, 107, 117, 127]],
      [[ 18,  28,  38,  48],
       [ 58,  68,  78,  88],
       [ 98, 108, 118, 128]],
      [[ 19,  29,  39,  49],
       [ 59,  69,  79,  89],
       [ 99, 109, 119, 129]]]]

       should be reshaped to:

       [[ 11,  12,  13,  21,  22,  23,  31,  32,  33,  41,  42,  43],
        [ 14,  15,  16,  24,  25,  26,  34,  35,  36,  44,  45,  46],
        [ 17,  18,  19,  27,  28,  29,  37,  38,  39,  47,  48,  49],
        [ 51,  52,  53,  61,  62,  63,  71,  72,  73,  81,  82,  83],
        [ 54,  55,  56,  64,  65,  66,  74,  75,  76,  84,  85,  86],
        [ 57,  58,  59,  67,  68,  69,  77,  78,  79,  87,  88,  89],
        [ 91,  92,  93, 101, 102, 103, 111, 112, 113, 121, 122, 123],
        [ 94,  95,  96, 104, 105, 106, 114, 115, 116, 124, 125, 126],
        [ 97,  98,  99, 107, 108, 109, 117, 118, 119, 127, 128, 129]]
    '''
    block_height, block_width, in_channels = array.shape[:3]
    rows = []

    max_element_count = section_height * int(image_width / MIN_SQUARE_SIZE)
    element_count = 0

    for i in range(in_channels):
      rows.append(array[:, :, i, :].reshape(block_height, -1, order='F'))
      element_count += block_height * in_channels * block_width

      if element_count >= max_element_count:
        break

    return np.vstack(rows)


  def _arrays_to_sections(self, arrays, section_height, image_width):
    '''
    input: unprocessed numpy arrays.
    returns: columns of the size that they will appear in the image, not scaled
             for display. That needs to wait until after variance is computed.
    '''
    sections = []
    section_area = section_height * image_width

    for array in arrays:
      if len(array.shape) == 4:
        section = self._conv_section(array, section_height, image_width)
      else:
        flattened_array = np.ravel(array)[:int(section_area / MIN_SQUARE_SIZE)]
        cell_count = np.prod(flattened_array.shape)
        cell_area = section_area / cell_count

        cell_side_length = floor(sqrt(cell_area))
        row_count = max(1, int(section_height / cell_side_length))
        col_count = int(cell_count / row_count)

        # Reshape the truncated array so that it has the same aspect ratio as
        # the section.

        # Truncate whatever remaining values there are that don't fit. Hopefully
        # it doesn't matter that the last few (< section count) aren't there.
        section = np.reshape(flattened_array[:row_count * col_count],
                             (row_count, col_count))

      sections.append(im_util.resize(section, section_height, image_width))

    self.sections_over_time.append(sections)

    if self.config['mode'] == 'variance':
      sections = self._sections_to_variance_sections(self.sections_over_time)

    return sections


  def _sections_to_variance_sections(self, sections_over_time):
    '''Computes the variance of corresponding sections over time.

    Returns:
      a list of np arrays.
    '''
    variance_sections = []

    for i in range(len(sections_over_time[0])):
      time_sections = [sections[i] for sections in sections_over_time]
      variance = np.var(time_sections, axis=0)
      variance_sections.append(variance)

    return variance_sections


  def _sections_to_image(self, sections):
    padding_size = 20

    sections = im_util.scale_sections(sections, self.config['scaling'])

    final_stack = [sections[0]]
    padding = np.ones((padding_size, IMAGE_WIDTH)) * 245

    for section in sections[1:]:
      final_stack.append(padding)
      final_stack.append(section)

    image_height = len(sections) * SECTION_HEIGHT +\
                   (padding_size * (len(sections) - 1))

    return im_util.resize(np.vstack(final_stack).astype(np.uint8),
                          image_height,
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


  def _save_section_info(self, arrays, sections):
    infos = []

    if self.config['values'] == 'trainable_variables':
      names = [x.name for x in tf.trainable_variables()]
    else:
      names = range(len(arrays))

    for array, section, name in zip(arrays, sections, names):
      info = {}

      info['name'] = name
      info['shape'] = str(array.shape)
      info['min'] = '{:.3e}'.format(section.min())
      info['mean'] = '{:.3e}'.format(section.mean())
      info['max'] = '{:.3e}'.format(section.max())
      info['range'] = '{:.3e}'.format(section.max() - section.min())

      infos.append(info)

    write_pickle(infos, '{}/{}'.format(self.logdir, SECTION_INFO_FILENAME))


  def build_frame(self, arrays):
    self._maybe_clear_deque()

    if self.config['values'] == 'trainable_variables':
      arrays = [self.SESSION.run(x) for x in tf.trainable_variables()]

    elif self.config['values'] == 'arrays':
      if arrays is None:
        arrays = [self.SESSION.run(x) for x in tf.trainable_variables()]
      else:
        arrays = arrays if isinstance(arrays, list) else [arrays]

    sections = self._arrays_to_sections(arrays, SECTION_HEIGHT, IMAGE_WIDTH)
    self._save_section_info(arrays, sections)
    final_image = self._sections_to_image(sections)


    return final_image

  def update(self, config):
    self.config = config
