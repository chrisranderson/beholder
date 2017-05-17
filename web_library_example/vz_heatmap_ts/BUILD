package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("@org_tensorflow//tensorflow/tensorboard:defs.bzl", "tensorboard_typescript_genrule")

licenses(["notice"])  # Apache 2.0

web_library(
    name = "vz_heatmap_ts",
    srcs = [
        "vz-heatmap-ts.html",
        "vz-heatmap-ts.js",
    ],
    path = "/vz-heatmap-ts",
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
        "demo/demo.js",
        "demo/index.html",
        "index.html",
    ],
    path = "/vz-heatmap-ts",
    deps = [
        ":vz_heatmap_ts",
        "@org_polymer",
        "@org_polymer_iron_component_page",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_webcomponentsjs",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports_d3v4:d3",
    ],
)

tensorboard_typescript_genrule(
    name = "ts",
    srcs = [
        "demo/demo.ts",
        "vz-heatmap-ts.ts",
    ],
    typings = [
        "@org_definitelytyped//:polymer.d.ts",
        "@org_definitelytyped//:webcomponents.js.d.ts",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports_d3v4:d3.d.ts",
    ],
)
