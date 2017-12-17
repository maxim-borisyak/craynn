import numpy as np

from lasagne import layers
from lasagne import regularization

from ..subnetworks import achain
from ..subnetworks import get_kernels_by_type


__all__ = [
  'Expression', 'Net', 'net',
  'get_input_layer'
]

def get_number_of_params(layer):
  try:
    return layer.num_params()
  except:
    return np.sum([
      np.prod(param.get_value(borrow=True).shape)
      for param in layer.get_params()
    ])

class Expression(object):
  def __init__(self, inputs, outputs, *args, **kwargs):
    """
    A base class for savable expressions.
    """
    self._args = args
    self._kwargs = kwargs

    self._snapshot_index = None
    self._dump_dir = None

    self.inputs = inputs
    self.outputs = outputs

  def __str__(self):
    args_str = ', '.join([str(arg) for arg in self._args])
    kwargs_str = ', '.join(['%s = %s' % (k, v) for k, v in self._kwargs.items()])

    hyperparam_str = (
      '(' + ', '.join([args_str, kwargs_str]) + ')'
      if len(self._args) > 0 or len(self._kwargs) > 0
      else ''
    )

    return "%s%s" % (
      str(type(self).__name__),
      hyperparam_str
    )

  def __repr__(self):
    return str(self)

  @property
  def weights(self):
    return layers.get_all_param_values(self.outputs)

  @weights.setter
  def weights(self, weights):
    layers.set_all_param_values(self.outputs, weights)

  def kernels(self, kernel_type):
    """
    Selects params (kernels) of given type.
    Might be useful for applying different regularization.

    :param kernel_type: type of the kernel.
    :return: list of params from layers that have `<kernel type>_kernel` method.
    """
    return get_kernels_by_type(self.outputs, kernel_type)

  def save(self, path):
    import pickle
    with open(path, 'w') as f:
      pickle.dump(self.weights, f)

  def load(self, path):
    import pickle
    with open(path, 'r') as f:
      params = pickle.load(f)

    self.weights = params

    return self

  def __call__(self, *args, **kwargs):
    external_inputs = [
      input for input in self.inputs if not hasattr(input, 'get_autoinput')
    ]
    auto_inputs = [
      input for input in self.inputs if hasattr(input, 'get_autoinput')
    ]
    external_substitutes = dict(zip(external_inputs, args))
    auto_substitutes = dict()

    for auto_input in auto_inputs:
      if auto_input.linked_layer is not None:
        linked = external_substitutes[auto_input.linked_layer]
      else:
        linked = None

      auto_substitutes[auto_input] = auto_input.get_autoinput(linked)

    substitutes = external_substitutes.copy()
    substitutes.update(auto_substitutes)

    return layers.get_output(self.outputs, inputs=substitutes, **kwargs)

  def reg_l1(self):
    return regularization.regularize_network_params(self.outputs, penalty=regularization.l1)

  def reg_l2(self):
    return regularization.regularize_network_params(self.outputs, penalty=regularization.l2)

  def description(self):
    def describe_layer(l):
      return '%s\n  output shape:%s\n  number of params: %s' % (l, l.output_shape, get_number_of_params(l))

    summary = '%s -> %s\ntotal number of params: %d' % (
      ' x '.join([ str(layers.get_output_shape(input)) for input in  self.inputs ]),
      ' x '.join([ str(layers.get_output_shape(output)) for output in self.outputs]),
      int(np.sum([ get_number_of_params(l) for l in layers.get_all_layers(self.outputs) ]))
    )
    layer_wise = '\n'.join([describe_layer(l) for l in layers.get_all_layers(self.outputs)])

    return '%s\n===========\n%s\n==========\n%s' % (str(self), summary, layer_wise)

  def total_number_of_parameters(self):
    return int(np.sum([get_number_of_params(l) for l in layers.get_all_layers(self.outputs)]))

  def params(self, **tags):
    return layers.get_all_params(self.outputs, **tags)

def is_shape(shape_or_layer):
  return hasattr(shape_or_layer, '__iter__') and all([ (type(s) is int or s is None) for s in shape_or_layer ])

def get_input_layer(shape_or_layer, index=None):
  if is_shape(shape_or_layer) :
    name = 'input' if index is None else 'input%d' % index
    return layers.InputLayer(shape=shape_or_layer, name=name)
  else:
    return shape_or_layer


class Net(Expression):
  """
  Just a simple class to produce Expressions in a generic manner.
  """
  def __init__(self, factory, inputs):
    ### either single layer instance or one shape
    if not hasattr(inputs, '__iter__') or is_shape(inputs):
      input_layer = get_input_layer(inputs)
      outputs = factory(input_layer)
      input_layers = [input_layer]
    else:
      input_layers = [
        get_input_layer(input, i)
        for i, input in enumerate(inputs)
      ]
      outputs = factory(input_layers)

    if not hasattr(outputs, '__iter__'):
      outputs = [outputs]

    super(Net, self).__init__(input_layers, outputs)


"""
Allows nice syntax:
```
  net([input1, input2, ...])(
    constructor
  )
```

or 

```
  net(input)(
    constructor
  )
```

for single input.
"""
net = lambda inputs: lambda *factory: Net(achain(*factory), inputs)