from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
from PIL import Image

import numpy as np
import tensorflow as tf
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import event_accumulator
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
    self._LOGDIR_ROOT = context.logdir
    self._PLUGIN_LOGDIR = pau.PluginDirectory(self._LOGDIR_ROOT,
                                              _PLUGIN_PREFIX_ROUTE)


  def get_plugin_apps(self):
    return {
        FRAME_ROUTE: self._serve_beholder_frame,
        '/tags': self._serve_tags
    }


  def is_active(self):
    return bool(self._MULTIPLEXER) and any(self._runs_and_tags().values())


  def _reload_beholder_event(self):
    # TODO: Hacky.
    pattern = '{}/events.out.tfevents*'.format(self._PLUGIN_LOGDIR)
    file_path = tf.gfile.Glob(pattern)[0]
    print('file_path', file_path)
    accumulator = self._MULTIPLEXER.GetAccumulator(RUN_NAME)

    with tf.errors.raise_exception_on_not_ok_status() as status:
      reader = tf.pywrap_tensorflow.PyRecordReader_New(
          tf.compat.as_bytes(file_path), 0, tf.compat.as_bytes(''), status)
      try:
        reader.GetNext(status)
      except (tf.errors.DataLossError, tf.errors.OutOfRangeError):
        print('Couldn\'t reload beholder frame.')
        return

    event = tf.Event()
    event.ParseFromString(reader.record())

    # event = next(event_file_loader.EventFileLoader(file).Load())
    accumulator._ProcessEvent(event)



  def _runs_and_tags(self):
    '''
    Returns runs and tags for this plugin.
    '''
    return {
        run_name: run_data[event_accumulator.TENSORS]
        for (run_name, run_data) in self._MULTIPLEXER.Runs().items()
        if (run_name == RUN_NAME)
    }


  @wrappers.Request.application
  def _serve_tags(self, request):
    runs_and_tags = self._runs_and_tags()
    return http_util.Respond(request, runs_and_tags, 'application/json')


  @wrappers.Request.application
  def _serve_beholder_frame(self, request):
    print('\nrequest', request)
    self._reload_beholder_event()
    tensor = self._MULTIPLEXER.Tensors(RUN_NAME, TAG_NAME)[0]
    array = tf.contrib.util.make_ndarray(tensor.tensor_proto).astype(np.uint8)
    print('array.mean()', array.mean())
    image = Image.fromarray(array, mode='L') # L: 8-bit grayscale
    bytes_buffer = io.BytesIO()
    image.save(bytes_buffer, 'PNG')
    return http_util.Respond(request,
                             bytes_buffer.getvalue(),
                             'image/png')
