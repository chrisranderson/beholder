from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import time

from google.protobuf import message
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin
from tensorboard.plugins.beholder import beholder

# TODO: will this cause problems elsewhere? Added because of broken pipe errors.
# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE, signal_handler)

TAG_NAME = beholder.TAG_NAME

_PLUGIN_PREFIX_ROUTE = beholder.PLUGIN_NAME
FRAME_ROUTE = '/beholder-frame'
CONFIG_ROUTE = '/change-config'
RUN_NAME = 'plugins/{}'.format(_PLUGIN_PREFIX_ROUTE)


class BeholderPlugin(base_plugin.TBPlugin):

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self, context):
    self._MULTIPLEXER = context.multiplexer
    plugin_logdir = pau.PluginDirectory(context.logdir,
                                        _PLUGIN_PREFIX_ROUTE)
    self._SUMMARY_PATH = '{}/{}'.format(plugin_logdir,
                                        beholder.SUMMARY_FILENAME)
    self._CONFIG_PATH = '{}/{}'.format(plugin_logdir,
                                       'config')
    self.most_recent_frame = np.zeros((beholder.IMAGE_HEIGHT,
                                       beholder.IMAGE_WIDTH))
    self.served_new = 0
    self.served_old = 0.0001

    self.FPS = 30

  def get_plugin_apps(self):
    return {
        FRAME_ROUTE: self._serve_beholder_frame,
        CONFIG_ROUTE: self._serve_change_config,
        '/tags': self._serve_tags
    }


  def is_active(self):
    return tf.gfile.Exists(self._SUMMARY_PATH)


  def _get_image_from_summary(self):
    try:
      with open(self._SUMMARY_PATH, 'rb') as file:
        summary_string = file.read()

        if not summary_string:
          raise message.DecodeError('Empty summary.')

      summary_proto = tf.Summary()
      summary_proto.ParseFromString(summary_string)

      tensor_proto = summary_proto.value[0].tensor

      frame = tf.contrib.util.make_ndarray(tensor_proto).astype(np.uint8)

      if np.all(frame == self.most_recent_frame):
        self.served_old += 1
      else:
        self.served_new += 1

      self.most_recent_frame = frame
      return frame

    # The message didn't decode properly - maybe halfway written at read time?
    # The message wasn't there - for instance, when a run restarted.
    except (message.DecodeError, IOError):
      self.served_old += 1
      return self.most_recent_frame


  @wrappers.Request.application
  def _serve_tags(self, request):
    if self.is_active:
      runs_and_tags = {RUN_NAME: {'tensors': [TAG_NAME]}}
    else:
      runs_and_tags = {}

    return http_util.Respond(request,
                             runs_and_tags,
                             'application/json')


  @wrappers.Request.application
  def _serve_change_config(self, request):
    config = request.form
    self.FPS = int(config['FPS'])

    with open(self._CONFIG_PATH, 'w') as file:
      print('server writing config', config)
      json_string = json.dumps(config)
      print('json_string', json_string)
      file.write(json_string)


    return http_util.Respond(request,
                             {'config': config},
                             'application/json')


  def _frame_generator(self):
    while True:
      # TODO: review
      if self.FPS == 0:
        continue
      else:
        time.sleep(1/self.FPS)

      array = self._get_image_from_summary()
      image = Image.fromarray(array, mode='L') # L: 8-bit grayscale
      bytes_buffer = io.BytesIO()
      image.save(bytes_buffer, 'PNG')
      image_bytes = bytes_buffer.getvalue()

      frame_text = b'--frame\r\n'
      content_type = b'Content-Type: image/png\r\n\r\n'
      response_content = frame_text + content_type + image_bytes + b'\r\n\r\n'

      yield response_content


  @wrappers.Request.application
  def _serve_beholder_frame(self, request):
    # print('percent new frames',
    #        self.served_new / (self.served_old + self.served_new))
    mimetype = 'multipart/x-mixed-replace; boundary=frame'
    return wrappers.Response(response=self._frame_generator(),
                             status=200,
                             mimetype=mimetype)
