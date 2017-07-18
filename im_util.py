from __future__ import division, print_function


import numpy as np
from PIL import Image, ImageDraw, ImageFont
np.set_printoptions(linewidth=99999)

font_path = "tensorboard/plugins/beholder/resources/roboto-mono.ttf"
FONT = ImageFont.truetype(font_path, 12)

def resize(nparray, height, width):
  image = Image.fromarray(nparray)
  return np.array(image.resize((width, height), Image.NEAREST))


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
