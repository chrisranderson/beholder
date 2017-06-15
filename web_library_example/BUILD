package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "web_library_example",
    srcs = ["web-library-example.html"],
    path = "/",
    deps = [
        "//web_library_example/vz_heatmap",
        "@org_polymer_font_roboto",
        "@org_polymer_webcomponentsjs",
    ],
)

tensorboard_html_binary(
    name = "index",
    input_path = "/web-library-example.html",
    output_path = "/index.html",
    deps = [":web_library_example"],
)
