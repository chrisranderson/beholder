import tensorflow as tf


def tensors_to_image_tensor(tensors, scaling_scope, image_height, image_width):

  global_min = tf.reduce_min([tf.reduce_min(tensor) for tensor in tensors])
  global_max = tf.reduce_max([tf.reduce_max(tensor) for tensor in tensors])
  column_width = (image_width / len(tensors))


  def reshape_tensor(tensor):
    tensor = tf.squeeze(tf.contrib.layers.flatten(tf.expand_dims(tensor, 0)))

    element_count = tf.to_float(tf.size(tensor))
    product = column_width * element_count
    columns = tf.ceil(tf.sqrt(product / image_height))
    rows = tf.floor(element_count / columns)

    rows = tf.to_int32(rows)
    columns = tf.to_int32(columns)

    # Truncate whatever remaining values there are that don't fit. Hopefully,
    # it doesn't matter that the last few (< column count) aren't there.
    return tf.reshape(tensor[:rows * columns], (1, rows, columns, 1))


  reshaped_tensors = [reshape_tensor(tensor) for tensor in tensors]
  image_scaled_tensors = [scale_for_display(tensor,
                                            scaling_scope,
                                            global_min,
                                            global_max)
                          for tensor in reshaped_tensors]
  final_tensors = [tf.squeeze(tf.image.resize_nearest_neighbor(
      tensor,
      [tf.to_int32(image_height), tf.to_int32(column_width)]
  ))
                   for tensor in image_scaled_tensors]


  # return tf.random_uniform((600, 800), minval=0, maxval=255)
  return tf.concat(final_tensors, axis=1)


def scale_for_display(tensor, scaling_scope, global_min, global_max):

  if scaling_scope == 'tensor':
    minimum = tf.reduce_min(tensor)
    maximum = tf.reduce_max(tensor - minimum)

  elif scaling_scope == 'network':
    minimum = global_min
    maximum = global_max

  tensor -= minimum
  return tensor * (255 / maximum)
