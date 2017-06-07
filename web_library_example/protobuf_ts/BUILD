package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_proto_library")
load("@org_tensorflow//tensorflow/tensorboard:vulcanize.bzl", "tensorboard_html_binary")
load("@org_tensorflow//tensorflow/tensorboard:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "protobuf_ts",
    srcs = [
        "demo.html",
        "demo.ts",
    ],
    clutz_entry_points = ["proto.web_library_example.protobuf_ts.Message"],
    path = "/protobuf-ts",
    deps = [":example_proto"],
)

tensorboard_html_binary(
    name = "devserver",
    input_path = "/protobuf-ts/demo.html",
    output_path = "/protobuf-ts/demo.html",
    deps = [":protobuf_ts"],
)

closure_js_proto_library(
    name = "example_proto",
    srcs = ["example.proto"],
)
