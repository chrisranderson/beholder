workspace(name = "io_github_jart_web_library_example")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "4be8a887f6f38f883236e77bb25c2da10d506f2bf1a8e5d785c0f35574c74ca4",
    strip_prefix = "rules_closure-aac19edc557aec9b603cd7ffe359401264ceff0d",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/aac19edc557aec9b603cd7ffe359401264ceff0d.tar.gz",  # 2017-05-10
        "https://github.com/bazelbuild/rules_closure/archive/aac19edc557aec9b603cd7ffe359401264ceff0d.tar.gz",
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

http_archive(
    name = "org_tensorflow",
    sha256 = "97ac28e1dec5120dade1de92cd8d2dddd7e50d0e077190b69ce6e270afde8a4a",
    strip_prefix = "tensorflow-319b6ce8c6c845318416745fba20281d03ebf5c9",
    urls = [
        "http://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/319b6ce8c6c845318416745fba20281d03ebf5c9.tar.gz",  # 2017-05-17
        "https://github.com/tensorflow/tensorflow/archive/319b6ce8c6c845318416745fba20281d03ebf5c9.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()
