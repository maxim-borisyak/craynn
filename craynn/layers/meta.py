from lasagne.layers import InputLayer
from lasagne.layers import Layer

"""
Some of the code was shamefully stolen from the lasagne library.
"""

"""
The MIT License (MIT)

Copyright (c) 2014-2015 Lasagne contributors

Lasagne uses a shared copyright model: each contributor holds copyright over
their contributions to Lasagne. The project versioning records all such
contribution and copyright details.
By contributing to the Lasagne repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__all__ = [
  'AutoInputLayer',
  'AutoUpdateLayer',
  'CloningMachine',
  'CloneLayer',
  'clone',
  'bound',
  'get_output'
]

class AutoInputLayer(InputLayer):
  linked_layer = None

  def get_autoinput(self, linked_input):
    raise NotImplementedError()

class AutoUpdateLayer(Layer):
  def get_autoupdate_for(self, input, **kwargs):
    raise NotImplementedError()

from difflib import get_close_matches
from warnings import warn

from collections import OrderedDict

import theano
import numpy as np

def get_output(layer_or_layers, inputs=None, return_auto_updates=False, **kwargs):
  """
  Computes the output of the network at one or more given layers.
  Optionally, you can define the input(s) to propagate through the network
  instead of using the input variable(s) associated with the network's
  input layer(s).
  Parameters
  ----------
  layer_or_layers : Layer or list
      the :class:`Layer` instance for which to compute the output
      expressions, or a list of :class:`Layer` instances.
  inputs : None, Theano expression, numpy array, or dict
      If None, uses the input variables associated with the
      :class:`InputLayer` instances.
      If a Theano expression, this defines the input for a single
      :class:`InputLayer` instance. Will throw a ValueError if there
      are multiple :class:`InputLayer` instances.
      If a numpy array, this will be wrapped as a Theano constant
      and used just like a Theano expression.
      If a dictionary, any :class:`Layer` instance (including the
      input layers) can be mapped to a Theano expression or numpy
      array to use instead of its regular output.
  Returns
  -------
  output : Theano expression or list
      the output of the given layer(s) for the given network input
  Notes
  -----
  Depending on your network architecture, `get_output([l1, l2])` may
  be crucially different from `[get_output(l1), get_output(l2)]`. Only
  the former ensures that the output expressions depend on the same
  intermediate expressions. For example, when `l1` and `l2` depend on
  a common dropout layer, the former will use the same dropout mask for
  both, while the latter will use two different dropout masks.
  """
  from lasagne.layers import get_all_layers
  from lasagne.layers.input import InputLayer
  from lasagne.layers.base import MergeLayer, Layer
  from lasagne import utils

  # check if the keys of the dictionary are valid
  if isinstance(inputs, dict):
    for input_key in inputs.keys():
      if (input_key is not None) and (not isinstance(input_key, Layer)):
        raise TypeError("The inputs dictionary keys must be"
                        " lasagne layers not %s." %
                        type(input_key))
  # track accepted kwargs used by get_output_for
  accepted_kwargs = {'deterministic'}

  autoupdates = OrderedDict()
  autoupdate_sources = OrderedDict()

  # obtain topological ordering of all layers the output layer(s) depend on
  treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
  all_layers = get_all_layers(layer_or_layers, treat_as_input)
  # initialize layer-to-expression mapping from all input layers
  all_outputs = dict((layer, layer.input_var)
                     for layer in all_layers
                     if isinstance(layer, InputLayer) and
                     layer not in treat_as_input)
  # update layer-to-expression mapping from given input(s), if any
  if isinstance(inputs, dict):
    all_outputs.update((layer, utils.as_theano_expression(expr))
                       for layer, expr in inputs.items())
  elif inputs is not None:
    if len(all_outputs) > 1:
      raise ValueError("get_output() was called with a single input "
                       "expression on a network with multiple input "
                       "layers. Please call it with a dictionary of "
                       "input expressions instead.")
    for input_layer in all_outputs:
      all_outputs[input_layer] = utils.as_theano_expression(inputs)
  # update layer-to-expression mapping by propagating the inputs
  for layer in all_layers:
    if layer not in all_outputs:
      try:
        if isinstance(layer, MergeLayer):
          layer_inputs = [all_outputs[input_layer]
                          for input_layer in layer.input_layers]
        else:
          layer_inputs = all_outputs[layer.input_layer]
      except KeyError:
        # one of the input_layer attributes must have been `None`
        raise ValueError("get_output() was called without giving an "
                         "input expression for the free-floating "
                         "layer %r. Please call it with a dictionary "
                         "mapping this layer to an input expression."
                         % layer)

      all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)

      if isinstance(layer, AutoUpdateLayer) and return_auto_updates:
        auto_upd = layer.get_autoupdate_for(layer_inputs, **kwargs)

        for k in auto_upd:
          if k in autoupdate_sources:
            raise ValueError(
              'layer %r tries to set auto update '
              'for parameter %r which has been already'
              'set by layer %r' % (layer, autoupdates[k], autoupdate_sources[k])
            )

          autoupdates[k] = auto_upd[k]
          autoupdate_sources[k] = layer

      try:
        accepted_kwargs |= set(utils.inspect_kwargs(
          layer.get_output_for))
      except TypeError:
        # If introspection is not possible, skip it
        pass
      accepted_kwargs |= set(layer.get_output_kwargs)
  unused_kwargs = set(kwargs.keys()) - accepted_kwargs
  if unused_kwargs:
    suggestions = []
    for kwarg in unused_kwargs:
      suggestion = get_close_matches(kwarg, accepted_kwargs)
      if suggestion:
        suggestions.append('%s (perhaps you meant %s)'
                           % (kwarg, suggestion[0]))
      else:
        suggestions.append(kwarg)
    warn("get_output() was called with unused kwargs:\n\t%s"
         % "\n\t".join(suggestions))
  # return the output(s) of the requested layer(s) only
  try:
    if return_auto_updates:
      return [all_outputs[layer] for layer in layer_or_layers], autoupdates
  except TypeError:
    if return_auto_updates:
      return all_outputs[layer_or_layers], autoupdates

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
    if original.name is None:
      name = 'clone of %s' % type(original).__name__
    else:
      name = 'clone of %s' % original.name

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

