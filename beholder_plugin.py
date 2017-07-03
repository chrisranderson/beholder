from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins import base_plugin

# TODO: external changes
_PLUGIN_PREFIX_ROUTE = event_accumulator.BEHOLDER

TAGS_ROUTE = '/tags'
BEHOLDER_ROUTE = '/beholder-frame'


class BeholderPlugin(base_plugin.TBPlugin):

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self, context):
    self._multiplexer = context._multiplexer


  def get_plugin_apps(self):
    return {
      BEHOLDER_ROUTE: self._serve_beholder_frame,
      TAGS_ROUTE: self._serve_tags
    }

  def is_active(self):
    return bool(self._multiplexer) and any(self._runs_and_tags().values())
    

  def _runs_and_tags(self):
    '''
    Returns runs and tags for this plugin.
    '''
    return {
      # TODO: external changes
      run_name: run_data[event_accumulator.BEHOLDER]
      for (run_name, run_data) in self._multiplexer.Runs().items()
      if event_accumulator.BEHOLDER in run_data
    }

  @wrappers.Request.application
  def _serve_tags(self, request):
    runs_and_tags = self._runs_and_tags()
    return http_util.Respond(request, runs_and_tags, 'application/json')   

  @wrappers.Request.application
  def _serve_beholder_frame(self, request):
    run = request.args.get('run')
    tag = request.args.get('tag')
    return http_util.Respond(request,
                             self._multiplexer.Beholders(run, tag),
                             'image/png')

