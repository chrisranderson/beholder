# Description:
# TensorBoard plugin for tensors and tensor variance for an entire graph.

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
  name = "image_util",
  srcs = ["image_util.py"],
)

py_library(
  name = "beholder",
  srcs = ["beholder.py"],
  deps = [
    ":image_util",
    "//tensorboard/backend/event_processing:plugin_asset_util",
  ],
)


py_library(
  name = "beholder_plugin",
  srcs = [
    "beholder_plugin.py",
  ],
  srcs_version = "PY2AND3",
  deps = [
    ":beholder",
    "//tensorboard/backend:http_util",
    "//tensorboard/backend/event_processing:plugin_asset_util",
    "//tensorboard/backend/event_processing:event_accumulator",
    "//tensorboard/plugins:base_plugin",
  ],
)
