from collections import deque

import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau
import tensorboard.plugins.beholder.image_util as image_util


PARAMETERS = 'parameters'
TENSOR = 'tensor'
NETWORK = 'network'

PLUGIN_NAME = 'beholder'
TAG_NAME = 'beholder-frame'

IMAGE_HEIGHT = 600.0
IMAGE_WIDTH = (IMAGE_HEIGHT * (4.0/3.0))

# TODO: should some of these methods be pulled out into something else?

class Beholder():

  # TODO: stuff from other plugins: multiplexer, context argument
  def __init__(
      self,
      session,
      logdir,
      variance_duration=5,
      scaling_scope=TENSOR):

    self.LOGDIR_ROOT = logdir
    self.PLUGIN_LOGDIR = pau.PluginDirectory(logdir, PLUGIN_NAME)
    self.SCALING_SCOPE = scaling_scope
    self.SESSION = session
    self.VARIANCE_DURATION = variance_duration
    # flush_secs is set so high because I will explicitly tell it when to write.
    self.WRITER = tf.summary.FileWriter(self.PLUGIN_LOGDIR,
                                        max_queue=1,
                                        flush_secs=9999999)

    # TODO: store the version with the most computation already done.
    self.tensors_over_time = deque([], variance_duration)
    self.summary_op = None


  def _get_mode(self):
    try:
      return pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'mode')
    except KeyError:
      tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)

      with open(self.PLUGIN_LOGDIR + '/mode', 'w') as mode_file:
        mode_file.write(PARAMETERS)

      return pau.RetrieveAsset(self.LOGDIR_ROOT, PLUGIN_NAME, 'mode')


  def _get_tensors(self, mode):

    def get_parameters():
      return tf.trainable_variables()

    mode_options = {
        # TODO: add a parameter that allows people to set their own function?
        PARAMETERS: get_parameters,
    }

    return mode_options[mode]()


  def write_summary(self):
    summary = self.SESSION.run(self.summary_op)

    # TODO: Hacky. Sometimes the file could be missing. How can we ensure there
    # is always exactly one complete file?
    files = tf.gfile.Glob('{}/events.out.tfevents*'.format(self.PLUGIN_LOGDIR))
    for file in files:
      tf.gfile.Remove(file)

    self.WRITER.reopen()
    self.WRITER.add_summary(summary)
    self.WRITER.flush()
    self.WRITER.close()

  # TODO: allow people to use their own image
  def update(self):
    mode = self._get_mode()

    if self.summary_op is not None:
      self.write_summary()
    else:
      # TODO: this graph will change when the mode changes. Right now, mode
      # changes are ignored.
      tensors = self._get_tensors(mode)
      image_tensor = image_util.tensors_to_image_tensor(tensors,
                                                        self.SCALING_SCOPE,
                                                        IMAGE_HEIGHT,
                                                        IMAGE_WIDTH)
      self.summary_op = tf.summary.tensor_summary(TAG_NAME, image_tensor)
      self.write_summary()
