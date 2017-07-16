import numpy as np
import tensorflow as tf

from tensorboard.plugins.beholder import im_util
from tensorboard.plugins.beholder.shared_config import SECTION_HEIGHT,\
  IMAGE_WIDTH

class ImUtilTest(tf.test.TestCase):

  def setUp(self):
    pass


  def test_conv_section(self):
    for _ in range(100):
      shape = np.random.randint(1, 6, 4)
      height, width, in_channel, out_channel = shape
      array = np.reshape(range(np.prod(shape)), shape)
      reshaped = im_util.conv_section(array, SECTION_HEIGHT, IMAGE_WIDTH)

      for in_number in range(in_channel):
        for out_number in range(out_channel):
          to_test = reshaped[in_number * height: in_number * height + height,
                             out_number * width: out_number * width + width]
          true = array[:, :, in_number, out_number]
          self.assertAllEqual(true, to_test)

if __name__ == '__main__':
  tf.test.main()
