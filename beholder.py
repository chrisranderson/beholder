from collections import deque
import time

import tensorflow as tf

from tensorboard.backend.event_processing import plugin_asset_util as pau

PLUGIN_NAME = 'beholder'

PARAMETERS = 'parameters'
TENSOR = 'tensor'
NETWORK = 'network'

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


  def _tensors_to_image_tensor(self, tensors):

    global_min = tf.reduce_min([tf.reduce_min(tensor) for tensor in tensors])
    global_max = tf.reduce_max([tf.reduce_max(tensor) for tensor in tensors])
    column_width = (IMAGE_WIDTH / len(tensors))

    def reshape_tensor(tensor):
      tensor = tf.squeeze(tf.contrib.layers.flatten(tf.expand_dims(tensor, 0)))

      element_count = tf.to_float(tf.size(tensor))
      product = column_width * element_count
      columns = tf.ceil(tf.sqrt(product / IMAGE_HEIGHT))
      rows = tf.floor(element_count / columns)

      rows = tf.to_int32(rows)
      columns = tf.to_int32(columns)

      # Truncate whatever remaining values there are that don't fit. Hopefully,
      # it doesn't matter that the last few (< column count) aren't there.
      return tf.reshape(tensor[:rows * columns], (1, rows, columns, 1))

    def scale_for_display(tensor):

      if self.SCALING_SCOPE == TENSOR:
        minimum = tf.reduce_min(tensor)
        maximum = tf.reduce_max(tensor - minimum)

      elif self.SCALING_SCOPE == NETWORK:
        minimum = global_min
        maximum = global_max

      tensor -= minimum
      return tensor * (255 / maximum)

    reshaped_tensors = [reshape_tensor(tensor) for tensor in tensors]
    image_scaled_tensors = [scale_for_display(tensor)
                            for tensor in reshaped_tensors]
    final_tensors = [tf.squeeze(tf.image.resize_nearest_neighbor(
        tensor,
        [tf.to_int32(IMAGE_HEIGHT), tf.to_int32(column_width)]
    ))
                     for tensor in image_scaled_tensors]

    return tf.concat(final_tensors, axis=1)

  def write_summary(self, image_tensor):
    d = time.time()
    # TODO: this gets slower and slower with each call.
    summary = self.SESSION.run(tf.summary.tensor_summary('beholder-frame-{}'.format(time.time()),
                                                         image_tensor))
    e = time.time()

    # TODO: there must be a better way to do this. Otherwise sometimes a file 
    #       isn't available. Also breaks if you want two images at once.
    
    files = tf.gfile.Glob('{}/events.out.tfevents*'.format(self.PLUGIN_LOGDIR))
    for file in files:
      tf.gfile.Remove(file)

    self.WRITER.reopen()
    self.WRITER.add_summary(summary)
    self.WRITER.flush()
    self.WRITER.close()
    print('Time to write summary: {}'.format(e - d))


  def update(self):
    mode = self._get_mode()
    tensors = self._get_tensors(mode)
    image_tensor = self._tensors_to_image_tensor(tensors)
    self.write_summary(image_tensor)
