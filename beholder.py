import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin

PLUGIN_NAME = 'beholder'

class Beholder(base_plugin.TBPlugin):

  # TODO: stuff from other plugins: multiplexer, context argument

  def __init__(
      self,
      session,
      logdir,
      mode='parameters',
      window_size=5,
      scaling='layer'):

    self.session = session
    self.logdir = logdir
    self.mode = mode
    self.window_size = window_size
    self.scaling = scaling

  def _get_mode(self):
    try:
      return pau.RetrieveAsset(self.logdir, PLUGIN_NAME, 'mode')
    except KeyError:
      directory = pau.PluginDirectory(self.logdir, PLUGIN_NAME)
      tf.gfile.MakeDirs(directory)

      with open(directory + '/mode', 'w') as mode_file:
        mode_file.write(self.mode)

      return pau.RetrieveAsset(self.logdir, PLUGIN_NAME, 'mode')

  def update(self):
    mode = self._get_mode()

  def get_plugin_apps(self):
    raise NotImplementedError()

  def is_active(self):
    raise NotImplementedError()

