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

import io
import lasagne

def get_hex_color(layer_class):
  COLORS = [
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

  layer_type = layer_class.__name__.lower()

  hashed = int(hash(layer_type)) % len(COLORS[0])

  if 'conv' in layer_type or issubclass(layer_class, lasagne.layers.conv.BaseConvLayer):
    return COLORS[0][hashed]

  if 'dense' in layer_type or issubclass(layer_class, lasagne.layers.DenseLayer):
    return COLORS[1][hashed]

  if 'input' is layer_type or issubclass(layer_class, lasagne.layers.InputLayer):
    return COLORS[2][hashed]

  if 'pool' in layer_type or layer_class.__name__ in lasagne.layers.pool.__all__:
    return COLORS[3][hashed]

  if layer_class.__name__ in lasagne.layers.recurrent.__all__:
    return COLORS[4][hashed]

  return COLORS[5][hashed]


def make_pydot_graph(layers, output_shape=True, verbose=False):
  """
  :parameters:
      - layers : list
          List of the layers, as obtained from lasagne.layers.get_all_layers
      - output_shape: (default `True`)
          If `True`, the output shape of each layer will be displayed.
      - verbose: (default `False`)
          If `True`, layer attributes like filter shape, stride, etc.
          will be displayed.
  :returns:
      - pydot_graph : PyDot object containing the graph
  """
  import pydotplus as pydot
  pydot_graph = pydot.Dot('Network', graph_type='digraph')
  pydot_nodes = {}
  pydot_edges = []
  for i, layer in enumerate(layers):
    if hasattr(layer, 'name') and getattr(layer, 'name') is not None:
      layer_type = '{0}\n{1}'.format(getattr(layer, 'name'), layer.__class__.__name__)
    else:
      layer_type = '{0}'.format(layer.__class__.__name__)

    key = repr(layer)
    label = layer_type
    color = get_hex_color(layer.__class__)
    if verbose:
      for attr in ['num_filters', 'num_units', 'ds',
                   'filter_shape', 'stride', 'strides', 'p']:
        if hasattr(layer, attr):
          label += '\n{0}: {1}'.format(attr, getattr(layer, attr))
      if hasattr(layer, 'nonlinearity'):
        try:
          nonlinearity = layer.nonlinearity.__name__
        except AttributeError:
          nonlinearity = layer.nonlinearity.__class__.__name__
        label += '\nnonlinearity: {0}'.format(nonlinearity)

    if output_shape:
      label += '\nOutput shape: {0}'.format(layer.output_shape)

    pydot_nodes[key] = pydot.Node(
      key, label=label, shape='record', fillcolor=color, style='filled')

    if hasattr(layer, 'input_layers'):
      for input_layer in layer.input_layers:
        pydot_edges.append([repr(input_layer), key])

    if hasattr(layer, 'input_layer'):
      pydot_edges.append([repr(layer.input_layer), key])

  for node in pydot_nodes.values():
    pydot_graph.add_node(node)

  for edges in pydot_edges:
    pydot_graph.add_edge(
      pydot.Edge(pydot_nodes[edges[0]], pydot_nodes[edges[1]]))
  return pydot_graph


def draw_to_file(layers_or_net, filename, **kwargs):
  """
  Draws a network diagram to a file
  :parameters:
      - layers : list or NeuralNet instance
          List of layers or the neural net to draw.
      - filename : string
          The filename to save output to
      - **kwargs: see docstring of make_pydot_graph for other options
  """

  from ..networks import Net

  if isinstance(layers_or_net, Net):
    layers = layers_or_net.layers
  else:
    layers = layers_or_net

  layers = lasagne.layers.get_all_layers(layers)
  dot = make_pydot_graph(layers, **kwargs)
  ext = filename[filename.rfind('.') + 1:]
  with io.open(filename, 'wb') as fid:
    fid.write(dot.create(format=ext))


def draw_to_notebook(layers_or_net, **kwargs):
  """
  Draws a network diagram in an IPython notebook
  :parameters:
      - layers : list or NeuralNet instance
          List of layers or the neural net to draw.
      - **kwargs : see the docstring of make_pydot_graph for other options
  """

  from ..networks import Net

  if isinstance(layers_or_net, Net):
    layers = layers_or_net.layers
  else:
    layers = layers_or_net

  from IPython.display import Image
  layers = (layers.get_all_layers() if hasattr(layers, 'get_all_layers')
            else layers)
  dot = make_pydot_graph(layers, **kwargs)
  return Image(dot.create_png())
