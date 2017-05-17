# Web Library Example [![Build Status](https://travis-ci.org/jart/web_library_example.svg?branch=master)](https://travis-ci.org/jart/web_library_example)

This repository demonstrates how to bootstrap a web application, like
TensorBoard, that uses Bazel, Polymer, TypeScript, D3, Plottable, ThreeJS, etc.

First you must [install] the latest Bazel. Then you may clone this repository
and cd into the directory.

To bring up the vz_heatmap documentation in a raw sources development server:

```sh
bazel run //web_library_example/vz_heatmap:index
```

Here is a TypeScript version of the same thing:

```sh
bazel run //web_library_example/vz_heatmap_ts:index
```

To bring up a development web server with the vulcanized HTML binary:

```sh
bazel run //web_library_example:index
```

You'll notice that every single import, script, and stylesheet was inlined into
one file. You can deploy it to production on a GCS bucket as follows:

```sh
bazel build //web_library_example:index
gsutil cp -a public-read -Z bazel-bin/web_library_example/index.html gs://my-bucket
```

To build everything:

```sh
bazel build //web_library_example/...
```

[install]: https://bazel.build/versions/master/docs/install.html
