from __future__ import division, print_function

import json
import time

import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau

from tensorboard.plugins.beholder import im_util
from tensorboard.plugins.beholder.visualizer import Visualizer
from tensorboard.plugins.beholder.shared_config import PLUGIN_NAME, TAG_NAME,\
  SUMMARY_FILENAME, default_config, INFO_HEIGHT, IMAGE_WIDTH


class Beholder():

  def __init__(self, session, logdir):
    self.visualizer = Visualizer(session)

    self.LOGDIR_ROOT = logdir
    self.PLUGIN_LOGDIR = pau.PluginDirectory(logdir, PLUGIN_NAME)
    self.SESSION = session

    self.frame_placeholder = None
    self.summary_op = None

    self.last_image_height = 0
    self.last_update_time = time.time()


  def _update_config(self):
    '''Reads the config file from disk or creates a new one.'''
    try:
      json_string = pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'config')
      config = json.loads(json_string)
    except (KeyError, ValueError):
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)

      with open(self.PLUGIN_LOGDIR + '/config', 'w') as config_file:
        config_file.write(json.dumps(default_config()))

      config = default_config()

    for key, value in config.items():
      try:
        config[key] = int(value)
      except ValueError:
        pass

    return config


  def _write_summary(self, frame):
    '''Writes the frame to disk as a tensor summary.'''
    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    with open(path, 'wb') as file:
      file.write(summary)


  def _get_final_image(self, config, arrays=None, frame=None):
    if config['values'] == 'frames':
      if frame is None:
        message = "A frame wasn't passed into the update function."
        final_image = im_util.text_image(INFO_HEIGHT, IMAGE_WIDTH, message)
      else:
        frame = frame() if callable(frame) else frame
        final_image = im_util.scale_image_for_display(frame)

    else:
      final_image = self.visualizer.build_frame(arrays)

    return final_image


  def _enough_time_has_passed(self, FPS):
    '''For limiting how often frames are computed.'''
    if FPS == 0:
      return False
    else:
      earliest_time = self.last_update_time + (1.0 / FPS)
      return time.time() >= earliest_time


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
    config = self._update_config()
    self.visualizer.update(config)

    if self._enough_time_has_passed(config['FPS']):
      self.last_update_time = time.time()

      final_image = self._get_final_image(config, arrays, frame)
      image_height, image_width = final_image.shape

      if self.summary_op is None or self.last_image_height != image_height:
        self.frame_placeholder = tf.placeholder(tf.uint8, [image_height,
                                                           image_width])
        self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                    self.frame_placeholder)
      self._write_summary(final_image)
      self.last_image_height = image_height

  ##############################################################################

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
