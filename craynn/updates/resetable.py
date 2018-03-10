"""
Optimizers implementations are based on ones from lasagne.
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


import numpy as np

import theano
import theano.tensor as T

from lasagne import *
from lasagne.updates import get_or_compute_grads

from collections import OrderedDict

__all__ = [
  'adam',
  'adamax'
]

def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8, scale_factor=None):
  """Adam updates
  Adam updates implemented as in [1]_.
  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float or symbolic scalar
      Learning rate
  beta1 : float or symbolic scalar
      Exponential decay rate for the first moment estimates.
  beta2 : float or symbolic scalar
      Exponential decay rate for the second moment estimates.
  epsilon : float or symbolic scalar
      Constant for numerical stability.
  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  OrderedDict
      A dictionary mapping each parameter to their default values.
  Notes
  -----
  The paper [1]_ includes an additional hyperparameter lambda. This is only
  needed to prove convergence of the algorithm and has no practical use
  (personal communication with the authors), it is therefore omitted here.
  References
  ----------
  .. [1] Kingma, Diederik, and Jimmy Ba (2014):
         Adam: A Method for Stochastic Optimization.
         arXiv preprint arXiv:1412.6980.
  """
  all_grads = get_or_compute_grads(loss_or_grads, params)
  t_prev = theano.shared(utils.floatX(0.))
  updates = OrderedDict()
  resets = OrderedDict()

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  t = t_prev + 1
  a_t = learning_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

  for param, g_t in zip(params, all_grads):
    value = param.get_value(borrow=True)
    m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable)
    v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable)

    resets[m_prev] = (
      m_prev / scale_factor
      if scale_factor is not None else
      T.zeros(value.shape, value.dtype)
    )

    resets[v_prev] = (
      v_prev * scale_factor
      if scale_factor is not None else
      T.zeros(value.shape, value.dtype)
    )

    if scale_factor is None:
      resets[t_prev] = 0.0

    m_t = beta1 * m_prev + (one - beta1) * g_t
    v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
    step = a_t * m_t / (T.sqrt(v_t) + epsilon)

    updates[m_prev] = m_t
    updates[v_prev] = v_t
    updates[param] = param - step

  updates[t_prev] = t

  return updates, resets

def adamax(loss_or_grads, params, learning_rate=0.002, beta1=0.9,
           beta2=0.95, epsilon=1e-8, scale_factor=None):
  """
  This version returns additional update to scale momentum params by the factor of `scale_factor`.
  Intended to be used for inner optimization in max-min problems.

  Adamax updates
  Adamax updates implemented as in [1]_. This is a variant of of the Adam
  algorithm based on the infinity norm.
  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float or symbolic scalar
      Learning rate
  beta1 : float or symbolic scalar
      Exponential decay rate for the first moment estimates.
  beta2 : float or symbolic scalar
      Exponential decay rate for the weighted infinity norm estimates.
  epsilon : float or symbolic scalar
      Constant for numerical stability.
  scale_factor: float or None
    Constant for scaling momentum parameters: first momentum is decreased by `scale_factor`,
    the second momentum is increased by the same factor.
    If None moments are set to zero.
  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  OrderedDict
      A dictionary mapping each parameter to its reset expression.

  References
  ----------
  .. [1] Kingma, Diederik, and Jimmy Ba (2014):
         Adam: A Method for Stochastic Optimization.
         arXiv preprint arXiv:1412.6980.
  """
  all_grads = get_or_compute_grads(loss_or_grads, params)
  t_prev = theano.shared(utils.floatX(0.))
  updates = OrderedDict()
  resets = OrderedDict()

  scale_factor = utils.floatX(scale_factor)

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  t = t_prev + 1
  a_t = learning_rate / (one - beta1 ** t)

  for param, g_t in zip(params, all_grads):
    value = param.get_value(borrow=True)
    m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable)
    u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                           broadcastable=param.broadcastable)

    m_t = beta1 * m_prev + (one - beta1) * g_t
    u_t = T.maximum(beta2 * u_prev, abs(g_t))
    step = a_t * m_t / (u_t + epsilon)

    updates[m_prev] = m_t
    updates[u_prev] = u_t
    updates[param] = param - step

    resets[m_prev] = (
      m_prev / scale_factor
      if scale_factor is not None else
      T.zeros(value.shape, value.dtype)
    )

    resets[u_prev] = (
      u_prev * scale_factor
      if scale_factor is not None else
      T.zeros(value.shape, value.dtype)
    )

    if scale_factor is None:
      resets[t_prev] = 0.0

  updates[t_prev] = t

  return updates, resets