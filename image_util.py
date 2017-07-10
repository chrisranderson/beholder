from math import ceil, floor, sqrt

import cv2
import numpy as np

def global_extrema(arrays):
  return min([x.min() for x in arrays]), max([x.max() for x in arrays])

def arrays_to_image(arrays, scaling_scope, image_height, image_width):
  global_min, global_max = global_extrema(arrays)
  column_width = (image_width / len(arrays))

  def reshape_array(array):
    array = np.ravel(array)

    element_count = np.prod(array.shape)
    columns = int(ceil(sqrt((column_width * element_count) / image_height)))
    rows = int(floor(element_count / columns))

    # Truncate whatever remaining values there are that don't fit. Hopefully,
    # it doesn't matter that the last few (< column count) aren't there.
    return np.reshape(array[:rows * columns], (rows, columns))

  reshaped_arrays = [reshape_array(array) for array in arrays]

  image_scaled_arrays = [scale_for_display(array,
                                           scaling_scope,
                                           global_min,
                                           global_max)
                         for array in reshaped_arrays]

  final_arrays = [cv2.resize(array,
                             (column_width, image_height),
                             interpolation=cv2.INTER_NEAREST)
                  for array in image_scaled_arrays]

  # Fixes little off-by-one errors after things are concatenated.
  return cv2.resize(np.hstack(final_arrays).astype(np.uint8),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST)


def scale_for_display(array, scaling_scope, global_min, global_max):

  if scaling_scope == 'layer':
    minimum = array.min()
    maximum = (array - minimum).max()

  elif scaling_scope == 'network':
    minimum = global_min
    maximum = global_max

  array -= minimum
  return array * (255 / maximum)
