package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_heatmap",
    srcs = ["vz-heatmap.html"],
    path = "/vz-heatmap",
    deps = [
        "@org_polymer",
        "@org_polymer_iron_resizable_behavior",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
    ],
)

web_library(
    name = "index",
    testonly = 1,
    srcs = [
        "demo/index.html",
        "index.html",
    ],
    path = "/vz-heatmap",
    deps = [
        ":vz_heatmap",
        "@org_polymer",
        "@org_polymer_iron_component_page",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_webcomponentsjs",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
    ],
)
