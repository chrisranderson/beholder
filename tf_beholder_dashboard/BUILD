package(default_visibility = ["//visibility:public"])

# load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "ts_web_library")
load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
  name = "tf_beholder_dashboard",
  deps = ["@org_polymer",
          "@org_tensorflow_tensorboard//tensorboard/components/tf_backend",
          "@org_tensorflow_tensorboard//tensorboard/components/tf_dashboard_common"],
  srcs = ["tf-beholder-dashboard.html"],
  path = "/tf-beholder-dashboard",
)