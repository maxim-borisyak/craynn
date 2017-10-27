from lasagne import *

from craynn import Expression

from ..subnetworks import achain


__all__ = [
  'Net', 'net',
  'get_input_layer'
]

def is_shape(shape_or_layer):
  return hasattr(shape_or_layer, '__iter__') and all([ (type(s) is int or s is None) for s in shape_or_layer ])

def get_input_layer(shape_or_layer, index=None):
  if is_shape(shape_or_layer) :
    name = 'input' if index is None else 'input%d' % index
    return layers.InputLayer(shape=shape_or_layer, name=name)
  else:
    return shape_or_layer


class Net(Expression):
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


net = lambda inputs: lambda *factory: Net(achain(*factory), inputs)