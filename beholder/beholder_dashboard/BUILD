load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "beholder_dashboard",
    srcs = [
        "beholder-dashboard.html",
        "beholder-video.html",
        "beholder-info.html",
    ],
    path = "/beholder-dashboard",
    deps = [
        "@org_tensorflow_tensorboard//tensorboard/components/tf_backend",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_card_heading",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_categorization_utils",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_color_scale",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_dashboard_common",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:d3",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:lodash",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:polymer",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_runs_selector",
        "@org_polymer_paper_radio_group",
        "@org_polymer_paper_button",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_tooltip",
    ],
)

ts_web_library(
    name = "demo",
    srcs = ["index.html"] + glob(["data/**"]),
    path = "/tf-beholder-dashboard",
    deps = [
        ":beholder-dashboard",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_backend",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:webcomponentsjs",
        "@org_polymer_iron_demo_helpers",
        "@org_polymer_paper_styles",
    ],
)
