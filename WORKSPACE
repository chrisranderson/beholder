workspace(name = "io_github_jart_web_library_example")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "bc41b80486413aaa551860fc37471dbc0666e1dbb5236fb6177cb83b0c105846",
    strip_prefix = "rules_closure-dec425a4ff3faf09a56c85d082e4eed05d8ce38f",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",  # 2017-06-02
        "https://github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")

closure_repositories()

http_archive(
    name = "org_tensorflow",
    sha256 = "c5181495da2b58070d5ee7634e321923c00da145f9cd0f7eb1b6c5f9dd5de368",
    strip_prefix = "tensorflow-f8e1cf8fa5fd244ae4cef738917675a90f2be301",
    urls = [
        "http://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/f8e1cf8fa5fd244ae4cef738917675a90f2be301.tar.gz",  # 2017-06-07
        "https://github.com/tensorflow/tensorflow/archive/f8e1cf8fa5fd244ae4cef738917675a90f2be301.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()
