from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import pickle
import time

from google.protobuf import message
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin
from beholder.shared_config import *

# TODO: will this cause problems elsewhere? Added because of broken pipe errors.
# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE, signal_handler)

FRAME_ROUTE = '/beholder-frame'
CONFIG_ROUTE = '/change-config'
SECTION_INFO_ROUTE = '/section-info'
RUN_NAME = 'plugins/{}'.format(PLUGIN_NAME)


class BeholderPlugin(base_plugin.TBPlugin):

  plugin_name = PLUGIN_NAME

  def __init__(self, context):
    self._MULTIPLEXER = context.multiplexer
    plugin_logdir = pau.PluginDirectory(context.logdir, PLUGIN_NAME)
    self._INFO_PATH = '{}/{}'.format(plugin_logdir, SECTION_INFO_FILENAME)
    self._SUMMARY_PATH = '{}/{}'.format(plugin_logdir, SUMMARY_FILENAME)
    self._CONFIG_PATH = '{}/{}'.format(plugin_logdir, CONFIG_FILENAME)
    self.most_recent_frame = np.zeros((SECTION_HEIGHT, IMAGE_WIDTH))
    self.served_new = 0
    self.served_old = 0.0001

    self.FPS = 10


  def get_plugin_apps(self):
    return {
        CONFIG_ROUTE: self._serve_change_config,
        FRAME_ROUTE: self._serve_beholder_frame,
        SECTION_INFO_ROUTE: self._serve_section_info,
        '/tags': self._serve_tags
    }


  def is_active(self):
    # TODO: bad idea :)
    return True
    # return tf.gfile.Exists(self._SUMMARY_PATH)


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
    config = {}

    for key, value in request.form.items():
      try:
        config[key] = int(value)
      except ValueError:
        if value == 'false':
          config[key] = False
        elif value == 'true':
          config[key] = True
        else:
          config[key] = value

    self.FPS = config['FPS']

    try:
      with open(self._CONFIG_PATH, 'w') as file:
        pickle.dump(config, file)

    except IOError:
      print('Could not write config file. Does the logdir exist?')

    return http_util.Respond(request, {'config': config}, 'application/json')

  @wrappers.Request.application
  def _serve_section_info(self, request):
    try:
      with open(self._INFO_PATH) as file:
        info = pickle.load(file)
    except (IOError, EOFError):
      info = []

    return http_util.Respond(request, info, 'application/json')

  def _frame_generator(self):
    while True:
      # TODO: review
      if self.FPS == 0:
        continue
      else:
        time.sleep(1/(self.FPS))

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
  def _serve_beholder_frame(self, request): # pylint: disable=unused-argument
    # print('percent new frames',
    #        self.served_new / (self.served_old + self.served_new))
    mimetype = 'multipart/x-mixed-replace; boundary=frame'
    return wrappers.Response(response=self._frame_generator(),
                             status=200,
                             mimetype=mimetype)
