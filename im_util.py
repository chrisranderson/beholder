from __future__ import division, print_function

from math import floor, sqrt
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

font_path = "tensorboard/plugins/beholder/resources/roboto-mono.ttf"
FONT = ImageFont.truetype(font_path, 48)


def resize(nparray, height, width):
  image = Image.fromarray(nparray)
  return np.array(image.resize((width, height), Image.NEAREST))


def global_extrema(arrays):
  return min([x.min() for x in arrays]), max([x.max() for x in arrays])


def conv_section(array, section_height, image_width):
  '''Reshape the conv layer so that the channels of the kernels stay together.

  Args:
    array: a numpy array of shape [a, b, c, d]

  Returns:
    a "section" - a numpy array of shape [section_height, image_width]. In this
    case, conv kernels maintain at least a little bit of spatial similarity.
  '''
  t1 = time.time()

  block_height, block_width, in_channels, out_channels = array.shape
  max_blocks = int((section_height * image_width) / (block_height*block_width))

  # Reshape the 4d array into a 2d array where kernel variables retain spatial
  # consistency. Individual kernel channels are now stacked horizontally. E.g.

  # x = np.array([
  #   [[[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 121], ],
  #    [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 122], ],
  #    [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 123], ]],
  #   [[[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 124], ],
  #    [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 125], ],
  #    [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 126], ]],
  #   [[[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 127], ],
  #    [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 128], ],
  #    [[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 129], ]],
  # ])

  # >>> x[:, :, 0, 0]
  # array([[1, 1, 1],
  #        [1, 1, 1]])

  # >>> x[:, :, 0, 1]
  # array([[2, 2, 2],
  #        [2, 2, 2]])

  # >>> x[:, :, 1, 0]
  # array([[5, 5, 5],
  #        [5, 5, 5]])

  # >>>x.reshape(2, 3*2*4, order='F')
  #[[11, 12, 13, 5, 5, 5, 2, 2, 2, 6, 6, 6, 3, 3, 3, 7, 7, 7, 4, 4, 4, 8, 8, 8],
  # [14, 15, 16, 5, 5, 5, 2, 2, 2, 6, 6, 6, 3, 3, 3, 7, 7, 7, 4, 4, 4, 8, 8, 8]]

  blocks_2d = array.reshape(block_height,
                            block_width * in_channels * out_channels,
                            order='F')[:, :max_blocks * block_width]

  # reorder the blocks from [1, 1, 1, 5, 5, 5] to [1, 1, 1, 2, 2, 2] etc.
  a = blocks_2d.reshape(block_height, block_width, -1, order='F')

  reordered_portions = []
  for x in range(block_height):
    portion = a[:, :, x::block_height].reshape(block_height, -1, order='F')
    reordered_portions.append(portion)

  blocks_2d = np.hstack(reordered_portions)

  block_count = in_channels * out_channels
  ratio = section_height / image_width

  # These commented out lines try to keep everything looking square.
  # These come from solving these equations:
  #   block_height * row_count / block_width * col_count == ratio
  #   row_count * col_count == block_count
  # row_count = int(sqrt(ratio * block_width * block_count / block_height)) + 1
  # col_count = int(block_count / row_count)
  row_count = in_channels
  col_count = out_channels # every column is a different filter
  row_width = col_count * block_width

  # TODO: is there a more efficient way to reshape these?
  # Take chunks out of the 2d matrix of filters and stack them vertically
  rows = []
  for row_number in range(row_count):
    slice_start = row_number * row_width
    slice_end = slice_start + (row_width)
    row = blocks_2d[:, slice_start:slice_end]

    if row.shape[1] == 0:
      continue

    if row.shape[1] != row_width:
      row = np.pad(row, [[0, 0], [0, row_width - row.shape[1]]], 'minimum')

    rows.append(row)

  if len(rows) == 1:
    section = rows[0]
  else:
    section = np.vstack(rows)
  t2 = time.time()

  print('t2-t1', t2-t1)
  return resize(section, section_height, image_width)


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
      sections.append(conv_section(array, section_height, image_width))
    else:
      flattened_array = np.ravel(array)[:section_area]
      cell_count = np.prod(flattened_array.shape)
      cell_area = section_area / cell_count

      cell_side_length = floor(sqrt(cell_area))
      row_count = max(1, int(section_height / cell_side_length))
      col_count = int(cell_count / row_count)

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
