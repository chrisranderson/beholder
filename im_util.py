from __future__ import division, print_function

from math import floor, sqrt

import numpy as np
from PIL import Image, ImageDraw, ImageFont
np.set_printoptions(linewidth=99999)

font_path = "tensorboard/plugins/beholder/resources/roboto-mono.ttf"
FONT = ImageFont.truetype(font_path, 12)

MIN_SQUARE_SIZE = 4


def resize(nparray, height, width):
  image = Image.fromarray(nparray)
  return np.array(image.resize((width, height), Image.NEAREST))


def global_extrema(arrays):
  return min([x.min() for x in arrays]), max([x.max() for x in arrays])


def conv_section(array, section_height, image_width):
  '''Reshape a rank 4 array to be rank 2, where each column of block_width is a
  filter, and each row of block height is an input channel. For example:

  [[[[ 11,  21,  31,  41],
     [ 51,  61,  71,  81],
     [ 91, 101, 111, 121]],
    [[ 12,  22,  32,  42],
     [ 52,  62,  72,  82],
     [ 92, 102, 112, 122]],
    [[ 13,  23,  33,  43],
     [ 53,  63,  73,  83],
     [ 93, 103, 113, 123]]],
   [[[ 14,  24,  34,  44],
     [ 54,  64,  74,  84],
     [ 94, 104, 114, 124]],
    [[ 15,  25,  35,  45],
     [ 55,  65,  75,  85],
     [ 95, 105, 115, 125]],
    [[ 16,  26,  36,  46],
     [ 56,  66,  76,  86],
     [ 96, 106, 116, 126]]],
   [[[ 17,  27,  37,  47],
     [ 57,  67,  77,  87],
     [ 97, 107, 117, 127]],
    [[ 18,  28,  38,  48],
     [ 58,  68,  78,  88],
     [ 98, 108, 118, 128]],
    [[ 19,  29,  39,  49],
     [ 59,  69,  79,  89],
     [ 99, 109, 119, 129]]]]

     should be reshaped to:

     [[ 11,  12,  13,  21,  22,  23,  31,  32,  33,  41,  42,  43],
      [ 14,  15,  16,  24,  25,  26,  34,  35,  36,  44,  45,  46],
      [ 17,  18,  19,  27,  28,  29,  37,  38,  39,  47,  48,  49],
      [ 51,  52,  53,  61,  62,  63,  71,  72,  73,  81,  82,  83],
      [ 54,  55,  56,  64,  65,  66,  74,  75,  76,  84,  85,  86],
      [ 57,  58,  59,  67,  68,  69,  77,  78,  79,  87,  88,  89],
      [ 91,  92,  93, 101, 102, 103, 111, 112, 113, 121, 122, 123],
      [ 94,  95,  96, 104, 105, 106, 114, 115, 116, 124, 125, 126],
      [ 97,  98,  99, 107, 108, 109, 117, 118, 119, 127, 128, 129]]
  '''
  block_height, block_width, in_channels = array.shape[:3]
  rows = []

  #
  max_element_count = section_height * int(image_width / MIN_SQUARE_SIZE)
  element_count = 0

  for i in range(in_channels):
    rows.append(array[:, :, i, :].reshape(block_height, -1, order='F'))
    element_count += block_height * in_channels * block_width

    if element_count >= max_element_count:
      break

  return np.vstack(rows)


def arrays_to_sections(arrays, section_height, image_width):
  '''
  input: unprocessed numpy arrays.
  returns: columns of the size that they will appear in the image, not scaled
           for display. That needs to wait until after variance is computed.
  '''
  sections = []
  section_area = section_height * image_width

  for array in arrays:
    if len(array.shape) == 4:
      section = conv_section(array, section_height, image_width)
    else:
      flattened_array = np.ravel(array)[:int(section_area / MIN_SQUARE_SIZE)]
      cell_count = np.prod(flattened_array.shape)
      cell_area = section_area / cell_count

      cell_side_length = floor(sqrt(cell_area))
      row_count = max(1, int(section_height / cell_side_length))
      col_count = int(cell_count / row_count)

      # Reshape the truncated array so that it has the same aspect ratio as the
      # section.

      # Truncate whatever remaining values there are that don't fit. Hopefully,
      # it doesn't matter that the last few (< section count) aren't there.
      section = np.reshape(flattened_array[:row_count * col_count],
                           (row_count, col_count))

    sections.append(resize(section, section_height, image_width))

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
  image = Image.new('L', (width, height), (245))
  draw = ImageDraw.Draw(image)
  draw.text((7, 17), text, font=FONT)
  return np.array(image).astype(np.uint8)


def scale_image_for_display(image, minimum=None, maximum=None):
  minimum = image.min() if minimum is None else minimum
  image -= minimum

  maximum = image.max() if maximum is None else maximum
  image *= 255 / maximum
  return image
