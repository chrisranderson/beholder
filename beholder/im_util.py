from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from beholder.file_system_tools import resources_path, read_pickle

def resize(nparray, height, width):
  image = Image.fromarray(nparray.astype(float))
  resized = np.array(image.resize((width, height), Image.NEAREST))
  return np.array(resized)


def global_extrema(arrays):
  return min([x.min() for x in arrays]), max([x.max() for x in arrays])


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


def scale_image_for_display(image, minimum=None, maximum=None):
  image = image.astype(float)

  minimum = image.min() if minimum is None else minimum
  image -= minimum

  maximum = image.max() if maximum is None else maximum

  if maximum == 0:
    return image
  else:
    image *= 255 / maximum
    return image.astype(np.uint8)


def pad_to_shape(array, shape, constant=245):
  padding = []

  for actual_dim, target_dim in zip(array.shape, shape):
    start_padding = 0
    end_padding = target_dim - actual_dim

    padding.append((start_padding, end_padding))

  return np.pad(array, padding, mode='constant', constant_values=constant)

# New matplotlib colormaps by Nathaniel J. Smith, Stefan van der Walt,
# and (in the case of viridis) Eric Firing.
#
# This file and the colormaps in it are released under the CC0 license /
# public domain dedication. We would appreciate credit if you use or
# redistribute these colormaps, but do not impose any legal restrictions.
#
# To the extent possible under law, the persons who associated CC0 with
# mpl-colormaps have waived all copyright and related or neighboring rights
# to mpl-colormaps.
#
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
colormaps = read_pickle('{}/colormaps.pkl'.format(resources_path()))
magma_data, inferno_data, plasma_data, viridis_data = colormaps

def apply_colormap(image, colormap='magma'):
  if colormap == 'grayscale':
    return image

  data_map = {
      'magma': magma_data,
      'inferno': inferno_data,
      'plasma': plasma_data,
      'viridis': viridis_data,
  }

  colormap_data = data_map[colormap]
  return (colormap_data[image]*255).astype(np.uint8)
