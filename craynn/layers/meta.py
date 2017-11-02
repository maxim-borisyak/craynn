from lasagne.layers import InputLayer
from lasagne.layers import Layer

__all__ = [
  'AutoInputLayer',
  'CloningMachine',
  'CloneLayer',
  'clone',
  'bound'
]

class AutoInputLayer(InputLayer):
  linked_layer = None

  def get_autoinput(self, linked_input):
    raise NotImplementedError()

class CloneLayer(Layer):
  """
  This layer redirects each call of `get_output_for` as `get_output_shape_for` to the original layer,
  effectively creating a copy of it.

  Important note: this layer does not have parameters on it own!
  Thus, `get_params` will always return an empty list.

  This is done to preserve common workflow:

  ```
  myNN = net(...)( <network description> )

  loss = ...

  upd = updates.super_optimizer(loss, myNN.params(trainable=True))
  ```

  Since CloneLayer does not have any params, optimizer will not receive several copies of the same parameter.
  """
  def __init__(self, incoming, original):
    self.original = original
    name = ('clone of %s' % original.name) if original.name is not None else None
    super(CloneLayer, self).__init__(incoming, name=name)

  def get_params(self, unwrap_shared=True, **tags):
    return []

  def get_output_for(self, input, **kwargs):
    return self.original.get_output_for(input, **kwargs)

  def get_output_shape_for(self, input_shape):
    return self.original.get_output_shape_for(input_shape)

class CloningMachine(object):
  """
  While layer model (e.g. `conv(num_filters=128)`) each time applied to a incoming layer yields
  a layer with independent parameters,
  `CloningMachine` ensures that applied multiple times, it returns the same layer (clone of the layer).

  Essentially, `CloningMachine` stores link to the first created layer,
  then produces `CloneLayer` on each subsequent call.

  Important note: only the first created layer has parameters,
  all subsequent layers have no parameters as they only redirect calls to the original layer!

  See also `craynn.layers.meta.CloneLayer`

  ```python
  op = conv(128)

  ### each call of `op` will produce
  ### independent layers
  net1 = net(...)(op)
  net2 = net(...)(op)

  X = T.ftensor4()

  f1 = theano.function([X], net1(X))
  f2 = theano.function([X], net2(X))

  data = ...

  ### assert will most likely fail.
  assert f1(data) == f2(data)

  ### but

  op = CloningMachine(conv(128))

  ### now op is a CloningMachine
  ### thus outputs the same layer each time it is called
  net1 = net(...)(op)
  net2 = net(...)(op)

  X = T.ftensor4()

  f1 = theano.function([X], net1(X))
  f2 = theano.function([X], net2(X))

  data = ...

  ### assert will most likely pass.
  assert f1(data) == f2(data)
  ```
  """
  def __init__(self, op):
    self.op = op
    self.original = None

  def __call__(self, incoming):
    if self.original is None:
      net = self.op(incoming)
      self.original = net
      return net
    else:
      return CloneLayer(incoming, self.original)

clone = lambda op: CloningMachine(op)
bound = lambda op: CloningMachine(op)

