# Beholder

![beholder demo video](https://raw.githubusercontent.com/chrisranderson/beholder/master/demo.gif)

Beholder is a TensorBoard plugin for visualizing frames of a video while your model trains. 

## Installation
As TensorBoard's (in progress) third party plugin system currently functions, you need to build TensorBoard from scratch to use this plugin.

1. [Install Bazel](https://docs.bazel.build/versions/master/install.html)
2. `pip install pillow`
3. `git clone https://github.com/tensorflow/tensorboard.git`
4. `cd tensorboard/tensorboard/plugins && git clone https://github.com/chrisranderson/beholder.git`
6. `cd beholder && bazel build demo && bazel run -- tensorboard --logdir="/tmp/beholder-demo`
7. 


## Usage

Before you begin training, initialize:

    visualizer = Beholder(session=sess,
                          logdir=LOG_DIRECTORY)

In your train loop, add:

    visualizer.update()

It allows you to visualize the parameters of your network and the variance of those parameters over time. You can also pass in a list of numpy arrays to `update()` (for example, gradient values returned from `sess.run([train_step, gradients])`.

Beholder just barely began development, and **it isn't currently in a state where it can easily be used by other people**. It will be, soon after the TensorBoard team makes some changes (see https://github.com/chrisranderson/beholder/issues/4).

Beholder will be in a very good state by August 16th. If you'd like to use it before then, post an issue and I'll help you out.
