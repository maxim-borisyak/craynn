### With a little shame stolen from nolearn

"""
Copyright (c) 2012-2015 Daniel Nouri

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from ..layers import DenseLayer, InputLayer, get_output_shape, get_layers
from craynn.layers.utils.inspect import get_number_of_params, get_total_number_of_params

__all__ = [
  'draw_to_file',
  'draw_to_notebook',
  'viz_all_params',
  'viz_params'
]

def full_class_name(clazz):
  return clazz.__module__ + '.' + clazz.__name__

def remove_craynn_layers_prefix(names):
  reduced_names = [
    ('.'.join(name.split('.')[3:]) if name.startswith('craynn.layers.') else name)
    for name in names
  ]
  if len(set(reduced_names)) < len(set(names)):
    return names
  else:
    return reduced_names

_color_set = [
  ### blue
  ('#a6cee3', '#1f78b4'),
  ### green
  ('#b2df8a', '#33a02c'),
  ### red
  ('#fb9a99', '#e31a1c'),
  ### orange
  ('#fdbf6f', '#ff7f00'),
  ### magenta
  ('#cab2d6', '#6a3d9a'),
  ### yellow/brown
  ('#ffff99', '#b15928'),
]

def _stable_hash(data):
  from hashlib import blake2b
  return int.from_bytes(blake2b(data.encode('utf-16be'), digest_size=8).digest(), byteorder='big')

def get_color(layer_class):
  layer_type = layer_class.__name__.lower()

  hashed = _stable_hash(layer_class.__name__) % len(_color_set[0])

  if 'conv' in layer_type:
    return _color_set[0][hashed]

  if issubclass(layer_class, DenseLayer) or 'dense' in layer_type:
    return _color_set[1][hashed]

  if issubclass(layer_class, InputLayer) or 'input' is layer_type:
    return _color_set[2][hashed]

  if 'pool' in layer_type:
    return _color_set[3][hashed]

  if 'recurrent' in layer_type:
    return _color_set[4][hashed]

  return _color_set[5][hashed]

def viz_params(**kwargs):
  def f(layer):
    param_info = get_number_of_params(layer, **kwargs).items()
    if len(param_info) > 0:
      return ','.join([ '%s: %d' % (k.name, v) for k, v in param_info ])
    else:
      return None

  return f

viz_all_params = lambda **kwargs: lambda layer: '%d' % get_total_number_of_params(layer, **kwargs)


def make_graph(layers, output_shape=('output shape', get_output_shape), **properties_to_display):
  import pydotplus as pydot

  additional_properties = list()

  for k in properties_to_display:
    property_spec = properties_to_display[k]

    if callable(property_spec):
      additional_properties.append((k, property_spec))
    elif property_spec is None:
      pass
    else:
      assert len(property_spec) == 2
      additional_properties.append(property_spec)

  additional_properties.append(output_shape)

  graph = pydot.Dot('network', graph_type='digraph')

  layer_indx = dict([
    (layer, 'node%d' % i) for i, layer in enumerate(layers)
  ])

  nodes = dict()


  layers_classes = remove_craynn_layers_prefix([
    layer.__class__.__name__ for layer in layers
  ])

  for layer, layer_class in zip(layers, layers_classes):
    if layer.name is None:
      info = [layer_class]
    else:
      info = [layer.name, layer_class]

    for prop_name, prop in additional_properties:
      try:
        if prop_name is not None:
          info.append('%s: %s' % (prop_name, prop(layer)))
        else:
          info.append('%s' % prop(layer))
      except Exception:
        pass

    nodes[layer] = pydot.Node(
      name=layer_indx[layer], shape='record',
      label='\n'.join(info),
      fillcolor=get_color(layer.__class__), style='filled'
    )

  for node in nodes.values():
    graph.add_node(node)

  for layer in layers:
    for incoming in getattr(layer, 'incomings', list()):
      graph.add_edge(pydot.Edge(nodes[incoming], nodes[layer]))

  return graph


def draw_to_file(layers_or_layer, path, **properties_to_display):
  graph = make_graph(get_layers(layers_or_layer), **properties_to_display)

  with open(path, 'wb') as f:
    f.write(
      graph.create(format='png')
    )


def draw_to_notebook(layers_or_layer, **properties_to_display):
  from IPython.display import Image

  graph = make_graph(get_layers(layers_or_layer), **properties_to_display)

  return Image(graph.create(format='png'))
