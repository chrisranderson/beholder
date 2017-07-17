import numpy as np
import tensorflow as tf

from tensorboard.plugins.beholder import im_util
from tensorboard.plugins.beholder.shared_config import SECTION_HEIGHT,\
  IMAGE_WIDTH

class ImUtilTest(tf.test.TestCase):

  def setUp(self):
    pass

  def test_conv_section(self):
    max_size = 5

    for height in range(1, max_size): # pylint:disable=too-many-nested-blocks
      for width in range(1, max_size):
        for in_channel in range(1, max_size):
          for out_channel in range(1, max_size):
            shape = [height, width, in_channel, out_channel]
            array = np.reshape(range(np.prod(shape)), shape)
            reshaped = im_util.conv_section(array, SECTION_HEIGHT, IMAGE_WIDTH)

            for in_number in range(in_channel):
              for out_number in range(out_channel):
                start_row = in_number * height
                start_col = out_number * width
                to_test = reshaped[start_row: start_row + height,
                                   start_col: start_col + width]
                true = array[:, :, in_number, out_number]
                self.assertAllEqual(true, to_test)

if __name__ == '__main__':
  tf.test.main()
