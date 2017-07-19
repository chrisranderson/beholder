# Description:
# TensorBoard plugin for tensors and tensor variance for an entire graph.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
  name = "im_util",
  data = ["resources"],
  srcs = ["im_util.py"],
)

py_test(
  name = "im_util_test",
  srcs = ["im_util_test.py", "shared_config.py"],
  srcs_version = "PY2AND3",
  deps = [
    ":im_util"
  ]
)

py_library(
  name = "visualizer",
  srcs = ["visualizer.py", "shared_config.py"],
  deps = [
    ":im_util"
  ],
)

py_library(
  name = "beholder",
  srcs = ["beholder.py", "shared_config.py", "video_writing.py"],
  deps = [
    ":im_util",
    ":visualizer",
    "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
  ],
)


py_library(
  name = "beholder_plugin",
  srcs = [
    "beholder_plugin.py",
    "shared_config.py"
  ],
  srcs_version = "PY2AND3",
  deps = [
    ":beholder",
    ":im_util",
    "@org_tensorflow_tensorboard//tensorboard/backend:http_util",
    "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
    "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:event_accumulator",
    "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
  ],
)
