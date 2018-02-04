from .meta import AutoUpdateLayer

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

import theano
import theano.tensor as T

from lasagne import init

class BatchNormLayer(AutoUpdateLayer):
  def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
               beta=init.Constant(0), gamma=init.Constant(1),
               mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
    super(BatchNormLayer, self).__init__(incoming, **kwargs)

    if axes == 'auto':
      # default: normalize over all but the second axis
      axes = (0,) + tuple(range(2, len(self.input_shape)))
    elif isinstance(axes, int):
      axes = (axes,)
    self.axes = axes

    self.epsilon = epsilon
    self.alpha = alpha

    # create parameters, ignoring all dimensions in axes
    shape = [size for axis, size in enumerate(self.input_shape)
             if axis not in self.axes]
    if any(size is None for size in shape):
      raise ValueError("BatchNormLayer needs specified input sizes for "
                       "all axes not normalized over.")
    if beta is None:
      self.beta = None
    else:
      self.beta = self.add_param(beta, shape, 'beta',
                                 trainable=True, regularizable=False)
    if gamma is None:
      self.gamma = None
    else:
      self.gamma = self.add_param(gamma, shape, 'gamma',
                                  trainable=True, regularizable=True)
    self.mean = self.add_param(mean, shape, 'mean',
                               trainable=False, regularizable=False)
    self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                  trainable=False, regularizable=False)


  def get_output_for(self, input, deterministic=False,
                     batch_norm_use_averages=None,
                     batch_norm_update_averages=None, **kwargs):
    input_mean = input.mean(self.axes)
    input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

    # Decide whether to use the stored averages or mini-batch statistics
    if batch_norm_use_averages is None:
      batch_norm_use_averages = deterministic
    use_averages = batch_norm_use_averages

    if use_averages:
      mean = self.mean
      inv_std = self.inv_std
    else:
      mean = input_mean
      inv_std = input_inv_std

    # Decide whether to update the stored averages
    if batch_norm_update_averages is None:
      batch_norm_update_averages = not deterministic
    update_averages = batch_norm_update_averages

    if update_averages:
      # Trick: To update the stored statistics, we create memory-aliased
      # clones of the stored statistics:
      running_mean = theano.clone(self.mean, share_inputs=False)
      running_inv_std = theano.clone(self.inv_std, share_inputs=False)
      # set a default update for them:
      running_mean.default_update = ((1 - self.alpha) * running_mean +
                                     self.alpha * input_mean)
      running_inv_std.default_update = ((1 - self.alpha) *
                                        running_inv_std +
                                        self.alpha * input_inv_std)
      # and make sure they end up in the graph without participating in
      # the computation (this way their default_update will be collected
      # and applied, but the computation will be optimized away):
      mean += 0 * running_mean
      inv_std += 0 * running_inv_std

    # prepare dimshuffle pattern inserting broadcastable axes as needed
    param_axes = iter(range(input.ndim - len(self.axes)))
    pattern = ['x' if input_axis in self.axes
               else next(param_axes)
               for input_axis in range(input.ndim)]

    # apply dimshuffle pattern to all parameters
    beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
    gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
    mean = mean.dimshuffle(pattern)
    inv_std = inv_std.dimshuffle(pattern)

    # normalize
    normalized = (input - mean) * (gamma * inv_std) + beta
    return normalized