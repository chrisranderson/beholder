load("@org_tensorflow_tensorboard//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")
load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "ts_web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:zipper.bzl", "tensorboard_zip_file")

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "tensorboard_x",
    srcs = ["main.py"],
    main = "main.py",
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow_tensorboard//tensorboard:util",
        "@org_tensorflow_tensorboard//tensorboard:version",
        "@org_tensorflow_tensorboard//tensorboard/backend:application",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:event_file_inspector",
        "@org_tensorflow_tensorboard//tensorboard/plugins/audio:audio_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/core:core_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/distribution:distributions_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/graph:graphs_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/histogram:histograms_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/image:images_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/profile:profile_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/projector:projector_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/scalar:scalars_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins/text:text_plugin",
        "@org_pocoo_werkzeug//:werkzeug",
        "//beholder:beholder_plugin",
    ],
)

ts_web_library(
    name = "index",
    srcs = ["index.html"],
    path = "/",
    deps = [
        "//beholder/beholder_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:webcomponentsjs",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_tensorboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/audio/tf_audio_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/distribution/tf_distribution_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/graph/tf_graph_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/histogram/tf_histogram_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/image/tf_image_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/profile/tf_profile_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/projector/vz_projector",
        "@org_tensorflow_tensorboard//tensorboard/plugins/scalar/tf_scalar_dashboard",
        "@org_tensorflow_tensorboard//tensorboard/plugins/text/tf_text_dashboard",
    ],
)


tensorboard_html_binary(
    name = "index_bin",
    input_path = "/index.html",
    output_path = "/index.html",
    deps = [":index"],
)

tensorboard_zip_file(
    name = "assets",
    deps = [":index_bin"],
)
