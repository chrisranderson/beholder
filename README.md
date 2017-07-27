# Beholder

![beholder demo video](https://raw.githubusercontent.com/chrisranderson/beholder/master/demo.gif)

Beholder is a TensorBoard plugin for viewing frames of a video while your model trains. It comes with tools to visualize the parameters of your network, visualize arbtirary tensors, or view frames that you've already created.

As TensorBoard's third party plugin system currently functions, you need to build a different version of TensorBoard from scratch to use this plugin.

## Build and run TensorBoard
1. [Install Bazel](https://docs.bazel.build/versions/master/install.html)
2. Clone the repo: `git clone https://github.com/chrisranderson/beholder.git`
3. `cd beholder`
4. Install Beholder: `pip install .`
4. Build TensorBoard (this will take a while): `bazel build beholder/tensorboard_x`
5. Run the newly built TensorBoard: `./bazel-bin/beholder/tensorboard_x/tensorboard_x --logdir=/tmp/beholder-demo`

## Run the demo
`bazel build beholder/demos/demo && ./bazel-bin/beholder/demos/demo/demo`

## Use Beholder in your own scripts
Before you begin training, create an instance of a Beholder:

```python
from beholder.beholder import Beholder
visualizer = Beholder(session=sess,
                      logdir=LOG_DIRECTORY)
```

In your train loop, add (to visualize `tf.trainable_variables()`:

```python
visualizer.update() # equivalent to visualizer.update(arrays=sess.run(tf.trainable_variables()))
```

To visualize arbitrary tensors:

```python
evaluated_tensors = session.run([var1, var2, var3])
visualizer.update(arrays=evaluated_tensors)
```

To watch frames that are already built:

```python
example_frame = np.random.randint(1, 255, (100, 100))
visualizer.update(frame=example_frame)
```

## Feedback

Please let me hear your thoughts/complaints/suggestions/success stories/unrelated banter. [Submit an issue](https://github.com/chrisranderson/beholder/issues/new), or [send me a direct message on Twitter](https://twitter.com/chrisdotio) (you don't need to follow me to send me a message).
