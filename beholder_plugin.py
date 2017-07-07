from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from PIL import Image

import numpy as np
import tensorflow as tf
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin
from tensorboard.plugins.beholder import beholder

TAG_NAME = beholder.TAG_NAME

# TODO: external changes
_PLUGIN_PREFIX_ROUTE = beholder.PLUGIN_NAME
FRAME_ROUTE = '/beholder-frame.png'
RUN_NAME = 'plugins/{}'.format(_PLUGIN_PREFIX_ROUTE)


class BeholderPlugin(base_plugin.TBPlugin):

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self, context):
    self._MULTIPLEXER = context.multiplexer
    plugin_logdir = pau.PluginDirectory(context.logdir,
                                        _PLUGIN_PREFIX_ROUTE)
    self._SUMMARY_PATH = '{}/{}'.format(plugin_logdir,
                                        beholder.SUMMARY_FILENAME)
    self.most_recent_frame = np.zeros((beholder.IMAGE_HEIGHT,
                                       beholder.IMAGE_WIDTH))
    self.served_new = 0
    self.served_old = 0.0001

  def get_plugin_apps(self):
    return {
        FRAME_ROUTE: self._serve_beholder_frame,
        '/tags': self._serve_tags
    }


  def is_active(self):
    return tf.gfile.Exists(self._SUMMARY_PATH)


  def _get_image_from_summary(self):
    try:
      with open(self._SUMMARY_PATH, 'rb') as file:
        summary_string = file.read()

      summary_proto = tf.Summary()
      summary_proto.ParseFromString(summary_string)
      tensor_proto = summary_proto.value[0].tensor
    # TODO: Probably not a great idea to catch all errors. :)
    except:
      self.served_old += 1
      return self.most_recent_frame

    frame = tf.contrib.util.make_ndarray(tensor_proto).astype(np.uint8)

    if np.all(frame == self.most_recent_frame):
      self.served_old += 1
    else:
      self.served_new += 1

    self.most_recent_frame = frame
    return frame


  @wrappers.Request.application
  def _serve_tags(self, request):
    if self.is_active:
      runs_and_tags = {RUN_NAME: {'tensors': [TAG_NAME]}}
    else:
      runs_and_tags = {}


  @wrappers.Request.application
  def _serve_beholder_frame(self, request):
    # print('percent new frames',
    #        self.served_new / (self.served_old + self.served_new))
    array = self._get_image_from_summary()
    image = Image.fromarray(array, mode='L') # L: 8-bit grayscale
    bytes_buffer = io.BytesIO()
    image.save(bytes_buffer, 'PNG')
    return http_util.Respond(request,
                             bytes_buffer.getvalue(),
                             'image/png')
