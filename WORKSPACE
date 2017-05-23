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
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")

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

new_http_archive(
    name = "org_angularjs_clutz",
    build_file = "//third_party:clutz.BUILD",
    sha256 = "3f95f6b89ead5deae245ea0c2a2ee9ee18c100e4adfafd4b9a6077ceaab048f8",
    strip_prefix = "clutz-6c190ee6a6584685c05781e34d1a5c4d58f2f103",
    urls = [
        "http://mirror.bazel.build/github.com/angular/clutz/archive/6c190ee6a6584685c05781e34d1a5c4d58f2f103.tar.gz",
        "https://github.com/angular/clutz/archive/6c190ee6a6584685c05781e34d1a5c4d58f2f103.tar.gz",
    ],
)

# This filegroup is meant to work around the fact that Clutz doesn't load the
# stuff in com.google.javascript.jscomp.DefaultExterns.
filegroup_external(
    name = "com_google_javascript_closure_compiler_externs",
    licenses = ["notice"],  # Apache 2.0
    sha256_urls_extract = {
        "0ee7b88ed2955b622eaa038bece283e28d0fb5abebfbb80871fc3d0353f0000b": [
            "http://mirror.bazel.build/github.com/google/closure-compiler/archive/v20170423.tar.gz",
            "https://github.com/google/closure-compiler/archive/v20170423.tar.gz",
        ],
    },
    strip_prefix = {"v20170423.tar.gz": "closure-compiler-20170423/externs"},
)
