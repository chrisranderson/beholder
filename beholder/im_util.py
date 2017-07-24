from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image

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
  minimum = image.min() if minimum is None else minimum
  image -= minimum

  maximum = image.max() if maximum is None else maximum
  image *= 255 / maximum
  return image
