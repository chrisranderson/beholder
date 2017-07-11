from math import ceil, floor, sqrt

import cv2
import numpy as np


def global_extrema(arrays):
  return min([x.min() for x in arrays]), max([x.max() for x in arrays])


def arrays_to_columns(arrays, image_height, image_width):
  '''
  Input: unprocessed numpy arrays.
  Returns: columns of the size that they will appear in the image, not scaled
           for display. That needs to wait until after variance is computed.
  '''
  column_width = (image_width / len(arrays))

  columns = []

  for array in arrays:
    flattened_array = np.ravel(array)

    element_count = np.prod(flattened_array.shape)
    col_count = int(ceil(sqrt((column_width * element_count) / image_height)))
    row_count = int(floor(element_count / col_count))

    # Truncate whatever remaining values there are that don't fit. Hopefully,
    # it doesn't matter that the last few (< column count) aren't there.
    columns.append(np.reshape(flattened_array[:row_count * col_count],
                              (row_count, col_count)))

  return [cv2.resize(column,
                     (column_width, image_height),
                     interpolation=cv2.INTER_NEAREST)
          for column in columns]


def scale_for_display(columns, scaling_scope):
  '''
  Input: unscaled columns.
  Returns: columns scaled to [0, 255]
  '''

  new_columns = []

  if scaling_scope == 'layer':
    for column in columns:
      column -= column.min()
      new_columns.append(column * (255 / column.max()))

  elif scaling_scope == 'network':
    global_min, global_max = global_extrema(columns)

    for column in columns:
      column -= global_min
      new_columns.append(column * (255 / global_max))

  return new_columns
