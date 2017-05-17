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

# TODO(jart): Update when #9981 is submitted.
#             https://github.com/tensorflow/tensorflow/pull/9981
http_archive(
    name = "org_tensorflow",
    sha256 = "20f2da72d1ed23d8dc5ebd719b2668a68af66f53f44ab215cb554767b39fc1af",
    strip_prefix = "tensorflow-visibility",
    urls = [
        "http://mirror.bazel.build/github.com/jart/tensorflow/archive/visibility.tar.gz",  # 2017-05-17
        "https://github.com/jart/tensorflow/archive/visibility.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()
