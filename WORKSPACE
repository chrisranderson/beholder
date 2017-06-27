workspace(name = "beholder")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e9e2538b1f7f27de73fa2914b7d2cb1ce2ac01d1abe8390cfe51fb2558ef8b27",
    strip_prefix = "rules_closure-4c559574447f90751f05155faba4f3344668f666",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/4c559574447f90751f05155faba4f3344668f666.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/4c559574447f90751f05155faba4f3344668f666.tar.gz",  # 2017-06-21
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")

closure_repositories()

http_archive(
    name = "org_tensorflow_tensorboard",
    sha256 = "d59b9cbdad1006742e6cbc15e9a03476f495c2820d1257358fb7106d5d8ba20f",
    strip_prefix = "tensorboard-866bb01b014f786c83462eed807587f3dd39de03",
    urls = [
        "http://mirror.bazel.build/github.com/tensorflow/tensorboard/archive/866bb01b014f786c83462eed807587f3dd39de03.tar.gz",
        "https://github.com/tensorflow/tensorboard/archive/866bb01b014f786c83462eed807587f3dd39de03.tar.gz",  # 2017-06-21
    ],
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")

tensorboard_workspace()
