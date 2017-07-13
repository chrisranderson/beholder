from __future__ import division, print_function

from math import floor, sqrt

import cv2
import numpy as np


def global_extrema(arrays):
  return min([x.min() for x in arrays]), max([x.max() for x in arrays])


def arrays_to_sections(arrays, section_height, image_width):
  '''
  input: unprocessed numpy arrays.
  returns: columns of the size that they will appear in the image, not scaled
           for display. That needs to wait until after variance is computed.
  '''
  sections = []
  section_area = section_height * image_width

  for array in arrays:
    flattened_array = np.ravel(array)[:section_area]
    cell_count = np.prod(flattened_array.shape)
    cell_area = section_area / cell_count

    cell_side_length = floor(sqrt(cell_area))
    row_count = max(1, int(floor(section_height / cell_side_length)))
    col_count = int(floor(cell_count / row_count))

    # Reshape the truncated array so that it has the same aspect ratio as the
    # section.

    # Truncate whatever remaining values there are that don't fit. Hopefully,
    # it doesn't matter that the last few (< section count) aren't there.
    reshaped = np.reshape(flattened_array[:row_count * col_count],
                               (row_count, col_count))
    resized = cv2.resize(reshaped,
                         (image_width, section_height),
                         interpolation=cv2.INTER_NEAREST)
    sections.append(resized)

  return sections


def scale_for_display(sections, scaling_scope):
  '''
  input: unscaled sections.
  returns: sections scaled to [0, 255]
  '''

  new_sections = []

  if scaling_scope == 'layer':
    for section in sections:
      section -= section.min()
      new_sections.append(section * (255 / section.max()))

  elif scaling_scope == 'network':
    global_min, global_max = global_extrema(sections)

    for section in sections:
      section -= global_min
      new_sections.append(section * (255 / global_max))

  return new_sections
