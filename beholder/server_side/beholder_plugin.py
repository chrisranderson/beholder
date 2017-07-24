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


class BeholderPlugin(base_plugin.TBPlugin):

  plugin_name = PLUGIN_NAME

  def __init__(self, context):
    self._MULTIPLEXER = context.multiplexer
    self.PLUGIN_LOGDIR = pau.PluginDirectory(context.logdir, PLUGIN_NAME)
    self.FPS = 10
    self.most_recent_frame = np.zeros((SECTION_HEIGHT, IMAGE_WIDTH))


  def get_plugin_apps(self):
    return {
        '/change-config': self._serve_change_config,
        '/beholder-frame': self._serve_beholder_frame,
        '/section-info': self._serve_section_info,
        '/tags': self._serve_tags
    }


  def is_active(self):
    return True


  def _fetch_current_frame(self):
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)

    try:
      frame = read_tensor_summary(path).astype(np.uint8)
      self.most_recent_frame = frame
      return frame

    except (message.DecodeError, IOError):
      return self.most_recent_frame


  @wrappers.Request.application
  def _serve_tags(self, request):
    if self.is_active:
      runs_and_tags = {
          'plugins/{}'.format(PLUGIN_NAME): {'tensors': [TAG_NAME]}
      }
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
      write_pickle(config, '{}/{}'.format(self.PLUGIN_LOGDIR, CONFIG_FILENAME))
    except IOError:
      print('Could not write config file. Does the logdir exist?')

    return http_util.Respond(request, {'config': config}, 'application/json')


  @wrappers.Request.application
  def _serve_section_info(self, request):
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SECTION_INFO_FILENAME)
    info = read_pickle(path, default=[])
    return http_util.Respond(request, info, 'application/json')


  def _frame_generator(self):
    while True:
      if self.FPS == 0:
        continue
      else:
        time.sleep(1/(self.FPS))

      array = self._fetch_current_frame()
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
