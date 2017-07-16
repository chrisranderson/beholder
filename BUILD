# Description:
# TensorBoard plugin for tensors and tensor variance for an entire graph.

package(default_visibility = ["//tensorboard:internal"])

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
  srcs = ["beholder.py", "shared_config.py"],
  deps = [
    ":im_util",
    ":visualizer",
    "//tensorboard/backend/event_processing:plugin_asset_util",
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
    "//tensorboard/backend:http_util",
    "//tensorboard/backend/event_processing:plugin_asset_util",
    "//tensorboard/backend/event_processing:event_accumulator",
    "//tensorboard/plugins:base_plugin",
  ],
)
