# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_DEFAULT_TYPINGS = [
    "@com_microsoft_typescript//:lib.es6.d.ts",
]

# TODO(jart): Delete this duplicated code once tensorboard_typescript_genrule()
#             starts using the --module es6 flag.
def tsc(name, srcs, typings=[], **kwargs):
  for src in srcs:
    if (src.startswith("/") or
        src.endswith(".d.ts") or
        not src.endswith(".ts")):
      fail("srcs must be typescript sources in same package")
  typings_out = [src[:-3] + ".d.ts" for src in srcs]
  inputs = _DEFAULT_TYPINGS + typings + srcs
  # These inputs are meant to work around a sandbox bug in Bazel. If we list
  # //third_party/javascript/node_modules/typescript:tsc under tools, then its
  # data attribute won't be considered when --genrule_strategy=sandboxed. See
  # https://github.com/bazelbuild/bazel/issues/1147 and its linked issues.
  data = [
      "@org_nodejs",
      "@com_microsoft_typescript",
  ]
  native.genrule(
      name = name,
      srcs = inputs + data,
      outs = [src[:-3] + ".js" for src in srcs] + typings_out,
      cmd = "$(location @com_microsoft_typescript//:tsc.sh)" +
            " --inlineSourceMap" +
            " --inlineSources" +
            # Do not follow triple slash references within typings.
            " --noResolve" +
            " --declaration" +
            " --module es6" +
            " --outDir $(@D) " +
            " ".join(["$(locations %s)" % i for i in inputs]),
      tools = ["@com_microsoft_typescript//:tsc.sh"],
      **kwargs
  )
  native.filegroup(
      name = name + "_typings",
      srcs = typings_out,
      **kwargs
  )
