package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_binary")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_library")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_proto_library")
load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "protobuf_js",
    srcs = [
        "demo.html",
        "demo_bin.js",
    ],
    path = "/protobuf-js",
)

closure_js_proto_library(
    name = "example_proto",
    srcs = ["example.proto"],
)

closure_js_library(
    name = "demo_lib",
    srcs = ["demo.js"],
    deps = [":example_proto"],
)

closure_js_binary(
    name = "demo_bin",
    deps = [":demo_lib"],
)
