import pickle

from google.protobuf import message
import tensorflow as tf

def write_file(contents, path, mode='wb'):
  with open(path, mode) as file:
    file.write(contents)


def read_tensor_summary(path):
  with open(path, 'rb') as file:
    summary_string = file.read()

  if not summary_string:
    raise message.DecodeError('Empty summary.')

  summary_proto = tf.Summary()
  summary_proto.ParseFromString(summary_string)
  tensor_proto = summary_proto.value[0].tensor
  array = tf.contrib.util.make_ndarray(tensor_proto)

  return array


def write_pickle(obj, path):
  with open(path, 'w') as file:
    pickle.dump(obj, file)


def read_pickle(path, default=None):
  try:
    with open(path) as file:
      result = pickle.load(file)

  except (IOError, EOFError):
    result = default

  return result
