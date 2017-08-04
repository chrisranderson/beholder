from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
from PIL import Image

from google.protobuf import message
import tensorflow as tf

def write_file(contents, path, mode='wb'):
  with open(path, mode) as new_file:
    new_file.write(contents)


def read_tensor_summary(path):
  print('Reading tensor summary.')
  with open(path, 'rb') as summary_file:
    summary_string = summary_file.read()

  if not summary_string:
    raise message.DecodeError('Empty summary.')

  summary_proto = tf.Summary()
  print('Parsing from string.')
  summary_proto.ParseFromString(summary_string)
  tensor_proto = summary_proto.value[0].tensor
  print('Making ndarray.')
  a = time.time()
  array = tf.contrib.util.make_ndarray(tensor_proto)
  b = time.time()
  print('b-a', b-a)
  print('Returning array.')
  return array


def write_pickle(obj, path):
  with open(path, 'wb') as new_file:
    pickle.dump(obj, new_file)


def read_pickle(path, default=None):
  try:
    with open(path, 'rb') as pickle_file:
      result = pickle.load(pickle_file)

  except (IOError, EOFError, ValueError):
    # TODO: log this somehow? Could swallow errors I don't intend.
    result = default

  return result

def get_image_relative_to_script(filename):
  script_directory = os.path.dirname(__file__)
  filename = os.path.join(script_directory, 'resources/{}'.format(filename))
  return np.array(Image.open(filename))
