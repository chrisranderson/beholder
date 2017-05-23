package(default_visibility = ["//visibility:public"])

load("//third_party:clutz.bzl", "clutz")
load("//third_party:tsc.bzl", "tsc")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_binary")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_library")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_proto_library")
load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "protobuf_ts",
    srcs = [
        "demo.html",
        "demo_bin.js",
    ],
    path = "/protobuf-ts",
)

closure_js_proto_library(
    name = "example_proto",
    srcs = ["example.proto"],
)

clutz(
    name = "example_proto_typings",
    entry_points = ["proto.web_library_example.protobuf_ts.Message"],
    deps = [":example_proto"],
)

tsc(
    name = "ts",
    srcs = ["demo.ts"],
    typings = [
        "example_proto_typings.d.ts",
        "@org_angularjs_clutz//:src/resources/closure.lib.d.ts",
    ],
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
