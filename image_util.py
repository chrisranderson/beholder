from __future__ import division, print_function

from math import floor, sqrt

import numpy as np
from PIL import Image, ImageDraw, ImageFont

font_path = "tensorboard/plugins/beholder/resources/roboto-mono.ttf"
FONT = ImageFont.truetype(font_path, 48)


def resize(nparray, height, width):
  image = Image.fromarray(nparray)
  return np.array(image.resize((width, height), Image.NEAREST))


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
    resized = resize(reshaped, section_height, image_width)
    sections.append(resized)

  return sections


def scale_sections(sections, scaling_scope):
  '''
  input: unscaled sections.
  returns: sections scaled to [0, 255]
  '''

  new_sections = []

  if scaling_scope == 'layer':
    for section in sections:
      new_sections.append(scale_image_for_display(section))

  elif scaling_scope == 'network':
    global_min, global_max = global_extrema(sections)

    for section in sections:
      new_sections.append(scale_image_for_display(section,
                                                  global_min,
                                                  global_max))
  return new_sections


def text_image(height, width, text):
  image = Image.new('L', (3*width, 3*height), (245))
  draw = ImageDraw.Draw(image)
  draw.text((20, 50), text, (33), font=FONT)
  image = image.resize((width, height), Image.ANTIALIAS)
  return np.array(image).astype(np.uint8)


def scale_image_for_display(image, minimum=None, maximum=None):
  minimum = image.min() if minimum is None else minimum
  image -= minimum

  maximum = image.max() if maximum is None else maximum
  image *= 255 / maximum
  return image
