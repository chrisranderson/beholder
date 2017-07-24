from __future__ import division, print_function

import time

import numpy as np
import tensorflow as tf

from file_system_tools import read_pickle, write_pickle, write_file
import im_util
from shared_config import PLUGIN_NAME, TAG_NAME, SUMMARY_FILENAME,\
  DEFAULT_CONFIG, CONFIG_FILENAME
import video_writing
from visualizer import Visualizer

class Beholder():

  def __init__(self, session, logdir):
    self.video_writer = None

    self.PLUGIN_LOGDIR = logdir + '/plugins/' + PLUGIN_NAME
    self.SESSION = session

    self.frame_placeholder = None
    self.summary_op = None

    self.last_image_height = 0
    self.last_update_time = time.time()
    self.previous_config = DEFAULT_CONFIG

    tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)
    write_pickle(DEFAULT_CONFIG, '{}/{}'.format(self.PLUGIN_LOGDIR,
                                                CONFIG_FILENAME))
    self.visualizer = Visualizer(session, self.PLUGIN_LOGDIR)


  def _get_config(self):
    '''Reads the config file from disk or creates a new one.'''
    filename = '{}/{}'.format(self.PLUGIN_LOGDIR, CONFIG_FILENAME)
    config = read_pickle(filename, default=self.previous_config)
    self.previous_config = config
    return config


  def _write_summary(self, frame):
    '''Writes the frame to disk as a tensor summary.'''
    summary = self.SESSION.run(self.summary_op, feed_dict={
        self.frame_placeholder: frame
    })
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)
    write_file(summary, path)


  def _get_final_image(self, config, arrays=None, frame=None):
    if config['values'] == 'frames':
      if frame is None:
        final_image = np.reshape(range(100*100), (100, 100))
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


  def _update_frame(self, arrays, frame, config):
    final_image = self._get_final_image(config, arrays, frame)
    image_height, image_width = final_image.shape

    if self.summary_op is None or self.last_image_height != image_height:
      self.frame_placeholder = tf.placeholder(tf.uint8, [image_height,
                                                         image_width])
      self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                  self.frame_placeholder)
    self._write_summary(final_image)
    self.last_image_height = image_height

    return final_image


  def _update_recording(self, frame, config):
    '''Adds a frame to the video using ffmpeg if possible. If not, writes
    individual frames as png files in a directory.
    '''
    is_recording, fps = config['is_recording'], config['FPS']
    filename = self.PLUGIN_LOGDIR + '/video-{}.mp4'.format(time.time())

    if is_recording:
      if self.video_writer is None or frame.shape != self.video_writer.size:
        try:
          self.video_writer = video_writing.FFMPEG_VideoWriter(filename,
                                                               frame.shape,
                                                               fps)
        except OSError:
          print('Something broke with ffmpeg. Saving frames to disk instead.')
          self.video_writer = video_writing.PNGWriter(self.PLUGIN_LOGDIR,
                                                      frame.shape)
      self.video_writer.write_frame(frame)
    elif not is_recording and self.video_writer is not None:
      self.video_writer.close()
      self.video_writer = None


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
    config = self._get_config()
    self.visualizer.update(config)

    if self._enough_time_has_passed(config['FPS']):
      self.last_update_time = time.time()

      final_image = self._update_frame(arrays, frame, config)
      self._update_recording(final_image, config)


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
