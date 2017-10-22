import numpy as np

__all__ = [
  'random', 'inf_random', 'random_seq', 'seq', 'inf_random_seq'
]

def random(n_samples, batch_size=128, n_batches=None, replace=True, priors=None):
  if n_batches is None:
    n_batches = n_samples // batch_size

  for i in range(n_batches):
    yield np.random.choice(n_samples, size=batch_size, replace=replace, p=priors)

def inf_random(n_samples, batch_size=128, replace=True, priors=None):
  while True:
      yield np.random.choice(n_samples, size=batch_size, replace=replace, p=priors)

def seq(n_samples, batch_size=128):
  indx = np.arange(n_samples)

  n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)

  for i in range(n_batches):
    i_from = i * batch_size
    i_to = i_from + batch_size
    yield indx[i_from:i_to]

def random_seq(n_samples, batch_size=128, allow_smaller=False):
  indx = np.random.permutation(n_samples)

  n_batches = n_samples // batch_size + (1 if (n_samples % batch_size != 0) and allow_smaller else 0)

  for i in range(n_batches):
    i_from = i * batch_size
    i_to = i_from + batch_size
    yield indx[i_from:i_to]

def inf_random_seq(n_samples, batch_size=128, allow_smaller=False):
  n_batches = n_samples // batch_size + (1 if (n_samples % batch_size != 0) and allow_smaller else 0)

  while True:
    indx = np.random.permutation(n_samples)

    for i in range(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]

def traverse(f, X, batch_size=32):
  for indx in seq(X.shape[0], batch_size=batch_size):
    yield f(X[indx])