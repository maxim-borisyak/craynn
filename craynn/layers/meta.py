from collections import OrderedDict

__all__ = [
  'Layer',
  'InputLayer',
  'FunctionalLayer',
  'layer',
  'get_output'
]

class Layer(object):
  def __init__(self, name=None):
    self._name = name

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

  @property
  def name(self):
    if self._name is None:
      return self.__class__.__name__
    else:
      return self._name

  def get_output_shape_for(self, *input_shapes):
    raise NotImplementedError

  def get_output_for(self, *args, **kwargs):
    raise NotImplementedError

class InputLayer(Layer):
  def __init__(self, shape, variable=None, name=None):
    self.shape = shape
    self.variable = variable

    super(InputLayer, self).__init__(name)

  def get_output_shape_for(self):
    return self.shape

  def get_output_for(self):
    if self.variable is None:
      raise ValueError('Input error was asked to return a value when no value was specified.')
    else:
      return self.variable

class FunctionalLayer(Layer):
  def __init__(self, *incomings, name=None):
    self.incomings = incomings
    self.params = OrderedDict()

    super(FunctionalLayer, self).__init__(name)

  def get_params(self):
    return self.params

class CustomLayer(FunctionalLayer):
  def __init__(self, f, shape_f, *incomings, name=None):
    super(CustomLayer, self).__init__(*incomings, name=name)
    self.f = f
    self.shape_f = shape_f

  def get_output_for(self, *args):
    return self.f(*args)

  def get_output_shape_for(self, *input_shapes):
    return self.shape_f(*input_shapes)

_shape_id = lambda *input_shapes: input_shapes[0]
layer = lambda f, shape_f=_shape_id, name=None: lambda *incomings: CustomLayer(f, shape_f, *incomings, name=name)

def propagate(f, layers, substitutes):
  print('Computing values for %s' % layers)
  known_results = dict(substitutes.items())

  stack = list()
  stack.extend(layers)

  while len(stack) > 0:
    current_layer = stack.pop()
    print('Computing value for %s:' % current_layer)

    if current_layer in known_results:
      print('  already in cache: value[%s] = %s;' % (current_layer, known_results[current_layer]))
      continue
    else:
      print(' cache miss %s;' % known_results)

    incomings = getattr(current_layer, 'incomings', list())

    unknown_dependencies = [
      incoming
      for incoming in incomings
      if incoming not in known_results
    ]

    if len(unknown_dependencies) == 0:
      print(' applying layer %s;' % current_layer)
      known_results[current_layer] = f(
        current_layer,
        [ known_results[incoming] for incoming in incomings ]
      )
    else:
      print('  querying %s;' % unknown_dependencies)
      stack.append(current_layer)
      stack.extend(unknown_dependencies)
  print()
  return known_results

def get_output(layers, substitutes=None, **kwargs):
  from .utils import apply_with_kwrags

  if substitutes is None:
    substitutes = dict()

  def f(layer : Layer, incomings):
    return apply_with_kwrags(layer.get_output_for, *incomings, **kwargs)

  results = propagate(f, layers, substitutes)

  return [ results[layer] for layer in layers ]





