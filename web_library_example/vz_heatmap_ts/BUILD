package(default_visibility = ["//visibility:public"])

load("@org_tensorflow//tensorflow/tensorboard:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "vz_heatmap_ts",
    srcs = [
        "vz-heatmap-ts.html",
        "vz-heatmap-ts.ts",
    ],
    path = "/vz-heatmap-ts",
    deps = [
        "@org_polymer_iron_resizable_behavior",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports:d3",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports:polymer",
    ],
)

ts_web_library(
    name = "index",
    testonly = 1,
    srcs = [
        "demo/demo.ts",
        "demo/index.html",
        "index.html",
    ],
    path = "/vz-heatmap-ts",
    deps = [
        ":vz_heatmap_ts",
        "@org_polymer_iron_component_page",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_webcomponentsjs",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports:d3",
        "@org_tensorflow//tensorflow/tensorboard/components/tf_imports:polymer",
    ],
)
