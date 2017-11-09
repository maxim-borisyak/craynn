import theano
import theano.tensor as T

from lasagne.updates import get_or_compute_grads

from theano.sandbox.rng_mrg import MRG_RandomStreams

from collections import OrderedDict

__all__ = [
  'noisy_gradients',
  'noisy'
]

def noisy(updates, std=1.0e-3, srng_or_seed=None):
  """
  Add Gaussian noise to the updates. Breaks any symmetries and correlation present in the network.
  The resulting updates are mixture of random walk and `updates`.

  Note that this function interferes with `updates` much strongly than `noisy_gradients`.

  ```
  updates = noisy(adamax)(loss, params)
  ```

  :param updates: a function that receive list of gradients and list of network parameters
    as first two arguments and returns update dictionary. All optimizers from `lasagne.updates` has such form.

  :param std: scalar, standard deviation for gradient noise.
  :param srng_or_seed: theano random stream instance or integer or None.
    If theano andom stream then it will be used to generate noise, otherwise
    `theano.sandbox.rng_mrg.MRG_RandomStreams` will be created with provided or default seed.
  :return: modified update function.
  """
  def u(loss_or_grads, params, *args, **kwargs):
    if type(srng_or_seed) is int:
      srng = MRG_RandomStreams(srng_or_seed)
    elif srng_or_seed is None:
      srng = MRG_RandomStreams()
    else:
      srng = srng_or_seed

    upd = updates(loss_or_grads, params, *args, **kwargs)

    noisy_updates = OrderedDict()

    for param, new_param in upd:
      noisy_updates[param] = new_param + srng.normal(size=param.shape, ndim=param.ndim, std=std)

    return noisy_updates

  return u

def noisy_gradients(updates, std=1.0e-3, srng_or_seed=None):
  """
  Adds a little bit of adventurous spirit to the optimizer by applying Gaussian noise to the gradients.
  The resulting updates are somewhat equivalent to updates on smaller batches,
  but breaks any symmetries and correlation present in the network.

  ```
  updates = noisy_gradients(adamax)(loss, params)
  ```

  :param updates: a function that receive list of gradients and list of network parameters
    as first two arguments. All optimizers from `lasagne.updates` has such form.

  :param std: scalar, standard deviation for gradient noise.
  :param srng_or_seed: theano random stream instance or integer or None.
    If theano andom stream then it will be used to generate noise, otherwise
    `theano.sandbox.rng_mrg.MRG_RandomStreams` will be created with provided or default seed.
  :return: modified update function.
  """
  def u(loss_or_grads, params, *args, **kwargs):
    grads = get_or_compute_grads(loss_or_grads, params)

    if type(srng_or_seed) is int:
      srng = MRG_RandomStreams(srng_or_seed)
    elif srng_or_seed is None:
      srng = MRG_RandomStreams()
    else:
      srng = srng_or_seed

    noisy_grads = [ g + srng.normal(size=g.shape, ndim=g.ndim, std=std) for g in grads ]

    return updates(noisy_grads, params, *args, **kwargs)

  return u


