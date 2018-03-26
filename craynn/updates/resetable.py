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

from lasagne.updates import get_or_compute_grads
from lasagne.updates import utils as lasagne_utils

from ..utils import zeros

from collections import OrderedDict

__all__ = [
  'rmsprop',
  'adam',
  'adamax',
  'amsgrad',
  'dummy_reset'
]

dummy_reset = lambda algo: lambda *args, scale_factor=None, **kwargs: (algo(*args, **kwargs), OrderedDict())

def momentum(loss_or_grads, params, learning_rate=1.0e-3, rho=0.9, scale_factor=None):
  grads = get_or_compute_grads(loss_or_grads, params)

  updates = OrderedDict()
  resets = OrderedDict()

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)

    updates[velocity] = rho * velocity + (one - rho) * grad
    updates[param] = param - learning_rate * velocity

    if scale_factor is None:
      resets[velocity] = zeros(velocity)
    else:
      resets[velocity] = velocity / scale_factor

  return updates, resets


def rmsprop(loss_or_grads, params, learning_rate=1.0e-3, rho=0.9, epsilon=1e-6, scale_factor=None):
  """RMSProp updates

  Scale learning rates by dividing with the moving average of the root mean
  squared (RMS) gradients. See [1]_ for further description.

  Parameters
  ----------
  loss_or_grads : symbolic expression or list of expressions
      A scalar loss expression, or a list of gradient expressions
  params : list of shared variables
      The variables to generate update expressions for
  learning_rate : float or symbolic scalar
      The learning rate controlling the size of update steps
  rho : float or symbolic scalar
      Gradient moving average decay factor
  epsilon : float or symbolic scalar
      Small value added for numerical stability

  Returns
  -------
  OrderedDict
      A dictionary mapping each parameter to its update expression

  Notes
  -----
  `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average
  fast.

  Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
  learning rate :math:`\\eta_t` is calculated as:

  .. math::
     r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
     \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

  References
  ----------
  .. [1] Tieleman, T. and Hinton, G. (2012):
         Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
         Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
  """
  grads = get_or_compute_grads(loss_or_grads, params)
  updates = OrderedDict()
  resets = OrderedDict()

  # Using theano constant to prevent upcasting of float32
  one = T.constant(1)

  for param, grad in zip(params, grads):
    value = param.get_value(borrow=True)
    accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                         broadcastable=param.broadcastable)
    accu_new = rho * accu + (one - rho) * grad ** 2
    updates[accu] = accu_new

    resets[accu] = (
      accu * scale_factor
      if scale_factor is not None else
      zeros(accu)
    )

    updates[param] = param - (learning_rate * grad /
                              T.sqrt(accu_new + epsilon))

  return updates, resets

def amsgrad(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
            beta2=0.999, epsilon=1e-8, scale_factor=None):
  """AMSGrad updates

  AMSGrad updates implemented as in [1]_.

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

  References
  ----------
  .. [1] https://openreview.net/forum?id=ryQu7f-RZ
  """
  all_grads = get_or_compute_grads(loss_or_grads, params)
  t_prev = theano.shared(lasagne_utils.floatX(0.))
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
    v_hat_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

    m_t = beta1 * m_prev + (one - beta1) * g_t
    v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
    v_hat_t = T.maximum(v_hat_prev, v_t)
    step = a_t * m_t / (T.sqrt(v_hat_t) + epsilon)

    updates[m_prev] = m_t
    updates[v_prev] = v_t
    updates[v_hat_prev] = v_hat_t

    resets[m_prev] = (
      m_prev / scale_factor
      if scale_factor is not None else
      zeros(m_prev)
    )
    resets[v_prev] = (
      v_prev / scale_factor
      if scale_factor is not None else
      zeros(v_prev)
    )
    resets[v_hat_prev] = (
      v_hat_prev / scale_factor
      if scale_factor is not None else
      zeros(v_hat_prev)
    )

    updates[param] = param - step

  updates[t_prev] = t

  if scale_factor is None:
    resets[t_prev] = 0.0

  return updates, resets

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
  t_prev = theano.shared(lasagne_utils.floatX(0.))
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
      zeros(m_prev)
    )

    resets[v_prev] = (
      v_prev * scale_factor
      if scale_factor is not None else
      zeros(v_prev)
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
  t_prev = theano.shared(lasagne_utils.floatX(0.))
  updates = OrderedDict()
  resets = OrderedDict()

  if scale_factor is not None:
    scale_factor = lasagne_utils.floatX(scale_factor)

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
      resets[t_prev] = 1.0

  updates[t_prev] = t

  return updates, resets