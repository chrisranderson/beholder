from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import time

from google.protobuf import message
import numpy as np
from PIL import Image
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin
from beholder.shared_config import PLUGIN_NAME, SECTION_HEIGHT, IMAGE_WIDTH,\
  SECTION_INFO_FILENAME, CONFIG_FILENAME, TAG_NAME, SUMMARY_FILENAME
from beholder.file_system_tools import read_tensor_summary, read_pickle,\
  write_pickle

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
      frame = read_tensor_summary(self._SUMMARY_PATH).astype(np.uint8)
      self.most_recent_frame = frame
      return frame

    # The message didn't decode properly - maybe halfway written at read time?
    # The message wasn't there - for instance, when a run restarted.
    except (message.DecodeError, IOError):
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
      write_pickle(config, self._CONFIG_PATH)
    except IOError:
      print('Could not write config file. Does the logdir exist?')

    return http_util.Respond(request, {'config': config}, 'application/json')

  @wrappers.Request.application
  def _serve_section_info(self, request):
    info = read_pickle(self._INFO_PATH, default=[])
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
    mimetype = 'multipart/x-mixed-replace; boundary=frame'
    return wrappers.Response(response=self._frame_generator(),
                             status=200,
                             mimetype=mimetype)
