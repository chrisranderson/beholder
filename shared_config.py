PLUGIN_NAME = 'beholder'

TAG_NAME = 'beholder-frame'
SUMMARY_FILENAME = 'frame.summary'

def default_config():
  return {
    'values': 'trainable_variables',
    'mode': 'variance',
    'scaling': 'layer',
    'window_size': 15,
    'FPS': 10
  }

INFO_HEIGHT = 40
SECTION_HEIGHT = 256
IMAGE_WIDTH = 1024 + 512
