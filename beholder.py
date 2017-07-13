from collections import deque
import json
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
import tensorboard.plugins.beholder.image_util as im_util

font_path = "tensorboard/plugins/beholder/resources/roboto.ttf"

CUSTOM = 'custom'
PARAMETERS = 'parameters'

CURRENT = 'current'
VARIANCE = 'variance'

SCALE_LAYER = 'layer'
SCALE_NETWORK = 'network'

PLUGIN_NAME = 'beholder'
TAG_NAME = 'beholder-frame'
SUMMARY_FILENAME = 'frame.summary'

SECTION_HEIGHT = 80
IMAGE_WIDTH = 1200

INFO_HEIGHT = 40
FONT = ImageFont.truetype(font_path, int(INFO_HEIGHT*1.2))

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

    self.config = dict(DEFAULT_CONFIG)
    self.old_config = dict(DEFAULT_CONFIG)

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


  def _update_config(self):

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

    self.config = config


  def _get_section_info_images(self, arrays, sections):
    images = []

    for variable, array, section in zip(tf.trainable_variables(),
                                        arrays,
                                        sections):
      minimum = section.min()
      maximum = section.max()
      template = '{:^30} | {:^30} min: {:^30.3e} max: {:^30.3e} range: {:^30.3e}'
      section_string = template.format(variable.name,
                                       array.shape,
                                       minimum,
                                       maximum,
                                       maximum - minimum)
      image = Image.new('L', (3*IMAGE_WIDTH, 3*INFO_HEIGHT), (245))
      draw = ImageDraw.Draw(image)
      draw.text((20, 60), section_string, (33), font=FONT)
      image = image.resize((IMAGE_WIDTH, INFO_HEIGHT), Image.ANTIALIAS)
      images.append(np.array(image).astype(np.uint8))

    return images


  def _get_display_frame(self, arrays):
    '''
    input: config and numpy arrays that will be displayed as an image.
    returns: a numpy array image ready to write to disk.
    '''
    scaling = self.config['scaling']

    sections = im_util.arrays_to_sections(arrays, SECTION_HEIGHT, IMAGE_WIDTH)
    self.frames_over_time.append(sections)

    if self.config['mode'] == VARIANCE:
      variance_sections = []

      for i in range(len(sections)):
        variance = np.var([sections[i] for sections in self.frames_over_time],
                          axis=0)
        variance_sections.append(variance)

      sections = variance_sections

    section_info_images = self._get_section_info_images(arrays, sections)
    scaled_sections = im_util.scale_for_display(sections, scaling)
    sections_with_info = []

    for i in range(len(arrays)):
      sections_with_info.append(section_info_images[i])
      sections_with_info.append(scaled_sections[i])

    return cv2.resize(np.vstack(sections_with_info).astype(np.uint8),
                      (IMAGE_WIDTH,
                       len(arrays) * (SECTION_HEIGHT + INFO_HEIGHT)),
                      interpolation=cv2.INTER_NEAREST)


  def _write_summary(self, frame):
    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    with open(path, 'wb') as file:
      file.write(summary)


  def _update_deque(self):

    if self.config['values'] != self.old_config['values'] or \
       self.config['mode'] != self.old_config['mode'] or \
       self.config['scaling'] != self.old_config['scaling']:
      self.frames_over_time.clear()

    self.old_config = self.config

    window_size = self.config['window_size']

    if window_size != self.frames_over_time.maxlen:
      self.frames_over_time = deque(self.frames_over_time, window_size)


  def _enough_time_has_passed(self):
    if self.config['FPS'] == 0:
      return False
    else:
      earliest_time = self.last_update_time + (1.0 / self.config['FPS'])
      return time.time() >= earliest_time


  def update(self, arrays=None, frame=None):
    self._update_config()
    values = self.config['values']

    if values != CUSTOM or (values == CUSTOM and not arrays):
      arrays = [self.SESSION.run(x) for x in tf.trainable_variables()]

    if self._enough_time_has_passed():
      self._update_deque()

      if frame is None:
        frame = self._get_display_frame(arrays)

      if self.summary_op is not None:
        self._write_summary(frame)
      else:
        image_height = len(arrays) * (SECTION_HEIGHT + INFO_HEIGHT)
        self.frame_placeholder = tf.placeholder(tf.float32,
                                                [image_height,
                                                 IMAGE_WIDTH])
        self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                    self.frame_placeholder)
        self._write_summary(frame)

      self.last_update_time = time.time()
