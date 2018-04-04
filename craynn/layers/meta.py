from collections import OrderedDict

__all__ = [
  'Layer',
  'InputLayer',
  'FunctionalLayer',
  'custom_layer',

  'get_output',
  'get_output_shape',

  'get_params',
  'get_all_params',
  'get_layers'
]

class Layer(object):
  def __init__(self, name=None):
    self.name = name

  def __str__(self):
    if self.name is None:
      return self.__class__.__name__
    else:
      return self.name

  def __repr__(self):
    return str(self)

  def get_output_shape_for(self, *input_shapes, **kwargs):
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

class CustomLayer(FunctionalLayer):
  def __init__(self, f, shape_f, *incomings, name=None):
    super(CustomLayer, self).__init__(*incomings, name=name)
    self.f = f
    self.shape_f = shape_f

  def get_output_for(self, *args):
    return self.f(*args)

  def get_output_shape_for(self, *input_shapes):
    return self.shape_f(*input_shapes)

_default_shape_f = lambda *input_shapes: input_shapes[0]
custom_layer = lambda f, shape_f=_default_shape_f, name=None: lambda *incomings: CustomLayer(f, shape_f, *incomings, name=name)

def propagate(f, layers, substitutes):
  known_results = OrderedDict(substitutes.items())

  stack = list()
  stack.extend(layers)

  while len(stack) > 0:
    current_layer = stack.pop()

    if current_layer in known_results:
      continue

    incomings = getattr(current_layer, 'incomings', list())

    unknown_dependencies = [
      incoming
      for incoming in incomings
      if incoming not in known_results
    ]

    if len(unknown_dependencies) == 0:
      known_results[current_layer] = f(
        current_layer,
        [ known_results[incoming] for incoming in incomings ]
      )
    else:
      stack.append(current_layer)
      stack.extend(unknown_dependencies)
  return known_results

def map_graph(f, layer_or_layers):
  """
  A wrapper over `propagate` for functions that does not depend on incomings values.
    Results are topologically ordered.

  :param f: `Layer -> value`
  :param layer_or_layers: layers (or a single layer) on which to compute `f` (including all dependencies).
  :return: topologically ordered outputs.
  """

  if isinstance(layer_or_layers, Layer):
    layers = [layer_or_layers]
  else:
    layers = layer_or_layers

  results = propagate(
    f=lambda layer, *args: f(layer),
    layers=layers,
    substitutes=dict()
  )

  return results.values()

def reduce_graph(operator, strict=False):
  """
    Wraps operator into `propagate`-operator.

  :param operator: `Layer` -> function
  :param strict: if `False` use `apply_with_kwargs` wrapper on `operator` which filters key word arguments before passing
    them into propagated function; otherwise, passes `**kwargs` directly to the propagated function.
  :return: a getter, function `(list of layer, substitution dictionary=None, **kwargs) -> value`
    that computes the operator output for `layers`.
  """
  def getter(layers_or_layer, substitutes=None, **kwargs):
    from .utils import apply_with_kwrags

    if isinstance(layers_or_layer, Layer):
      layers = [layers_or_layer]
    else:
      layers = layers_or_layer

    if substitutes is None:
      substitutes = dict()
    if strict:
      wrapped_operator = lambda layer, incomings: apply_with_kwrags(operator(layer), *incomings, **kwargs)
    else:
      wrapped_operator = lambda layer, incomings: operator(layer)(*incomings, **kwargs)

    results = propagate(wrapped_operator, layers, substitutes)

    if isinstance(layers_or_layer, Layer):
      return results[layers_or_layer]
    else:
      return [results[layer] for layer in layers]

  return getter

get_output = reduce_graph(lambda layer: layer.get_output_for, strict=False)
get_output_shape = reduce_graph(lambda layer: layer.get_output_shape_for, strict=False)

get_layers = lambda layers_or_layer: map_graph(lambda layer: layer, layers_or_layer)

def get_params(layer: FunctionalLayer, **properties):
  if not hasattr(layer, 'params'):
    return []

  return [
    param
    for param in layer.params.keys()
    if all([
      (v == (k in layer.params[param]) or v is None)
      for k, v in properties.items()
    ])
  ]

def get_all_params(layer_or_layers, **properties):
  collected_params = map_graph(lambda layer: get_params(layer, **properties), layer_or_layers)
  return [ param for params in collected_params for param in params ]



