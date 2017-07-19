workspace(name = "beholder_plugin_root")

################################################################################
# CLOSURE RULES - Build rules and libraries for JavaScript development
#
# NOTE: SHA should match what's in TensorBoard's WORKSPACE file.
# NOTE: All the projects dependeded upon in this file use highly
#       available redundant URLs. They are strongly recommended because
#       they hedge against GitHub outages and allow Bazel's downloader
#       to guarantee high performance and 99.9% reliability. That means
#       practically zero build flakes on CI systems, without needing to
#       configure an HTTP_PROXY.

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

# Inherit external repositories defined by Closure Rules.
closure_repositories()

################################################################################
# GO RULES - Build rules and libraries for Go development
#
# NOTE: TensorBoard does not require Go rules; they are a transitive
#       dependency of rules_webtesting.
# NOTE: SHA should match what's in TensorBoard's WORKSPACE file.

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "f7e42a4c1f9f31abff9b2bdee6fe4db18bc373287b7e07a5b844446e561e67e2",
    strip_prefix = "rules_go-4c9a52aba0b59511c5646af88d2f93a9c0193647",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_go/archive/4c9a52aba0b59511c5646af88d2f93a9c0193647.tar.gz",  # 2017-05-05
        "https://github.com/bazelbuild/rules_go/archive/4c9a52aba0b59511c5646af88d2f93a9c0193647.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:def.bzl", "go_repositories")

# Inherit external repositories defined by Go Rules.
go_repositories()

################################################################################
# WEBTESTING RULES - Build rules and libraries for Go development
#
# NOTE: SHA should match what's in TensorBoard's WORKSPACE file.
# NOTE: Some external repositories are omitted because they were already
#       defined by closure_repositories().

http_archive(
    name = "io_bazel_rules_webtesting",
    sha256 = "bb278df2afe88ed01490e4b25e2c048d453a518cb77d4795f6232a10fbae6c1f",
    strip_prefix = "rules_webtesting-dc0530015f201c2707085deba93ad210e89e6d18",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_webtesting/archive/dc0530015f201c2707085deba93ad210e89e6d18.tar.gz",  # 2017-05-10
        "https://github.com/bazelbuild/rules_webtesting/archive/dc0530015f201c2707085deba93ad210e89e6d18.tar.gz",
    ],
)

load("@io_bazel_rules_webtesting//web:repositories.bzl", "browser_repositories", "web_test_repositories")

web_test_repositories(
    omit_com_google_code_findbugs_jsr305 = True,
    omit_com_google_code_gson = True,
    omit_com_google_errorprone_error_prone_annotations = True,
    omit_com_google_guava = True,
    omit_junit = True,
    omit_org_hamcrest_core = True,
)

################################################################################
# TENSORBOARD - Framework for visualizing machines learning
#
# NOTE: If the need should arise to patch TensorBoard's codebase, then
#       git clone it to local disk and use local_repository() instead of
#       http_archive(). This should be a temporary measure until a pull
#       request can be merged upstream. It is an anti-pattern to
#       check-in a WORKSPACE file that uses local_repository() since,
#       unlike http_archive(), it isn't automated. If upstreaming a
#       change takes too long, then consider checking in a change where
#       http_archive() points to the forked repository.

http_archive(
    name = "org_tensorflow_tensorboard",
    sha256 = "0c70b3356f6600c06f2930a74d01642b9fd9688187d27b374b3da60833800008",
    strip_prefix = "tensorboard-ca813e6a0083896463e587779db96ad2aa545b2e",
    urls = [
        "http://mirror.bazel.build/github.com/jart/tensorboard/archive/ca813e6a0083896463e587779db96ad2aa545b2e.tar.gz",
        "https://github.com/jart/tensorboard/archive/ca813e6a0083896463e587779db96ad2aa545b2e.tar.gz",  # 2017-07-18
    ],
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")

# Inherit external repositories defined by Closure Rules.
tensorboard_workspace()
