from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from tensorboard.backend import application

import tensorflow as tf
from werkzeug import test as werkzeug_test
from werkzeug import wrappers

from beholder.tensorboard_x.main import get_plugins

URL_PREFIX = 'data/plugin/beholder'

class MainTest(tf.test.TestCase):

  def setUp(self):
    app = application.standard_tensorboard_wsgi(
        '/tmp/beholder-demo',
        True,
        5,
        get_plugins()
    )
    self.server = werkzeug_test.Client(app, wrappers.BaseResponse)

  def _make_url(self, path):
    return URL_PREFIX + '/' + path

  def _post(self, path, data):
    response = self.server.post(self._make_url(path), data=data)
    self.assertEqual(200, response.status_code)
    return json.loads(response.get_data().decode('utf-8'))

  def _get_json(self, path):
    path = self._make_url(path)
    response = self.server.get(path)
    self.assertEqual(200, response.status_code)
    self.assertEqual('application/json', response.headers.get('Content-Type'))
    return json.loads(response.get_data().decode('utf-8'))

  def test_section_info(self):
    response = self._get_json('section-info')
    info = response[0]
    self.assertIn('range', info)

  def test_change_config(self):
    response = self._post('change-config', data={
        'values': 'trainable_variables',
        'mode': 'variance',
        'scaling': 'layer',
        'window_size': 15,
        'FPS': 10,
        'is_recording': 'false'
    })
    self.assertIn('window_size', response['config'])

  def test_beholder_frame(self):
    response = self.server.get(self._make_url('beholder-frame'))
    self.assertEqual(200, response.status_code)



if __name__ == '__main__':
  tf.test.main()
