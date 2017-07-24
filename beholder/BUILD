# Description:
# TensorBoard plugin for tensors and tensor variance for an entire graph.

package(default_visibility = ["//visibility:public"])
licenses(["notice"])  # Apache 2.0
exports_files(["LICENSE"])

py_library(
  name = "im_util",
  data = ["resources"],
  srcs = ["im_util.py"],
  srcs_version = "PY2AND3",
)

py_library(
  name = "visualizer",
  srcs = ["visualizer.py", "shared_config.py"],
  deps = [
    ":im_util"
  ],
  srcs_version = "PY2AND3",
)

py_library(
  name = "file_system_tools",
  srcs = ["file_system_tools.py"],
  srcs_version = "PY2AND3",
)

py_library(
  name = "beholder",
  srcs = ["beholder.py", "shared_config.py", "video_writing.py"],
  deps = [
    ":im_util",
    ":visualizer",
    ":file_system_tools",
    "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
  ],
  srcs_version = "PY2AND3",
)
