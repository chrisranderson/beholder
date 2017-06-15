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
    name = "org_tensorflow_tensorboard",
    sha256 = "b793efe5536b06debcfadfa9ce7e774cadf654e5e9d52f6570ac11060d62e3a7",
    strip_prefix = "tensorboard-7b3c93ca9b6aea715cc349dc10fb151c11c70e01",
    urls = [
        "http://mirror.bazel.build/github.com/tensorflow/tensorboard/archive/7b3c93ca9b6aea715cc349dc10fb151c11c70e01.tar.gz",  # 2017-06-14
        "https://github.com/tensorflow/tensorboard/archive/7b3c93ca9b6aea715cc349dc10fb151c11c70e01.tar.gz",
    ],
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")

tensorboard_workspace()
