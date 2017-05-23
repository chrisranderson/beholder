# Copyright 2017 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build definitions for Closure JavaScript libraries."""

load("@io_bazel_rules_closure//closure/private:defs.bzl",
     "JS_FILE_TYPE",
     "collect_js",
     "unfurl")

def _clutz(ctx):
  deps = unfurl(ctx.attr.deps, provider="closure_js_library")
  js = collect_js(ctx, deps)
  srcs = set(JS_FILE_TYPE.filter(ctx.files._browser_externs)) | js.srcs
  args = ["-o", ctx.outputs.typing.path]
  for src in srcs:
    args.append(src.path)
  if ctx.attr.entry_points:
    args.append("--closure_entry_points")
    args.extend(ctx.attr.entry_points)
  ctx.action(
      inputs=list(srcs),
      outputs=[ctx.outputs.typing],
      executable=ctx.executable._clutz,
      arguments=args,
      mnemonic="Clutz",
      progress_message="Running Clutz on %d JS files" % len(srcs))
  return struct(files=set([ctx.outputs.typing]))

clutz = rule(
    implementation=_clutz,
    attrs={
        "entry_points": attr.string_list(),
        "deps": attr.label_list(
            providers=["closure_js_library"], mandatory=True),
        "_browser_externs": attr.label(
            default=Label("@com_google_javascript_closure_compiler_externs"),
            allow_files=True),
        "_clutz": attr.label(
            default=Label("@org_angularjs_clutz//:clutz"),
            executable=True,
            cfg="host"),
    },
    outputs={
        "typing": "%{name}.d.ts",
    })
