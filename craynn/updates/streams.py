import numpy as np

import pyximport
pyximport.install()

import os
import os.path as osp
import itertools

import threading
from Queue import Queue

from crayimage.imgutils import slice

__all__ = [
  'random',
  'random_seq',
  'seq',
  'inf_random_seq',
  'hdf5', 'hdf5_from_disk',
  'np_from_disk',
  'sampling',
  'binned',
  'traverse'
]

def random(n_samples, batch_size=128, n_batches=None, replace=True, priors=None):
  if n_batches is None:
    n_batches = n_samples / batch_size

  for i in xrange(n_batches):
    yield np.random.choice(n_samples, size=batch_size, replace=replace, p=priors)

def seq(n_samples, batch_size=128):
  indx = np.arange(n_samples)

  n_batches = n_samples / batch_size + (1 if n_samples % batch_size != 0 else 0)

  for i in xrange(n_batches):
    i_from = i * batch_size
    i_to = i_from + batch_size
    yield indx[i_from:i_to]

def random_seq(n_samples, batch_size=128, allow_smaller=False):
  indx = np.random.permutation(n_samples)

  n_batches = n_samples / batch_size + (1 if (n_samples % batch_size != 0) and allow_smaller else 0)

  for i in xrange(n_batches):
    i_from = i * batch_size
    i_to = i_from + batch_size
    yield indx[i_from:i_to]

def inf_random_seq(n_samples, batch_size=128, allow_smaller=False):
  n_batches = n_samples / batch_size + (1 if (n_samples % batch_size != 0) and allow_smaller else 0)

  while True:
    indx = np.random.permutation(n_samples)

    for i in xrange(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]

def traverse(f, X, batch_size=1024):
  return np.vstack([
    f(X[indx])
    for indx in seq(X.shape[0], batch_size=batch_size)
  ])

def traverse_image(f, img, window = 40, step = 20, batch_size=32):
  patches = slice(img, window = window, step = step)
  patches_shape = patches.shape[:2]

  return traverse(f, patches, batch_size=batch_size).reshape(patches_shape + (-1, ))

def binned(target_statistics, batch_size, n_batches, n_bins=64):
  hist, bins = np.histogram(target_statistics, bins=n_bins)
  indx = np.argsort(target_statistics)
  indicies_categories = np.array_split(indx, np.cumsum(hist)[:-1])
  n_samples = target_statistics.shape[0]

  per_category = batch_size / n_bins

  weight_correction = (n_bins * np.float64(hist) / n_samples).astype('float32')
  wc = np.repeat(weight_correction, per_category)

  for i in xrange(n_batches):
    sample = [
      np.random.choice(ind, size=per_category, replace=True)
      for ind in indicies_categories
    ]

    yield np.hstack(sample), wc

def hdf5_batch_worker(path, out_queue, batch_sizes):
  import h5py
  import itertools

  f = h5py.File(path, mode='r')

  n_bins = len([ k for k in  f.keys() if k.startswith('bin_') ])

  datasets = [
    f['bin_%d' % i] for i in range(n_bins)
  ]

  if type(batch_sizes) in [long, int]:
    batch_sizes = [batch_sizes] * len(datasets)

  indxes_stream = itertools.izip(*[
    inf_random_seq(n_samples=dataset.shape[0], batch_size=batch_size)
    for dataset, batch_size in zip(datasets, batch_sizes)
  ])

  for indxes in indxes_stream:
    batch = np.vstack([
      ds[np.sort(ind).tolist()]
      for ind, ds in zip(indxes, datasets)
    ])

    out_queue.put(batch, block=True)

def hdf5(path, batch_sizes=8):
  import h5py
  import itertools

  f = h5py.File(path, mode='r')

  n_bins = len([ k for k in  f.keys() if k.startswith('bin_') ])

  datasets = [
    f['bin_%d' % i] for i in range(n_bins)
  ]

  if type(batch_sizes) in [long, int]:
    batch_sizes = [batch_sizes] * len(datasets)

  indxes_stream = itertools.izip(*[
    inf_random_seq(n_samples=dataset.shape[0], batch_size=batch_size)
    for dataset, batch_size in zip(datasets, batch_sizes)
  ])

  for indxes in indxes_stream:
    batch = np.vstack([
      ds[np.sort(ind).tolist()]
      for ind, ds in zip(indxes, datasets)
    ])

    yield batch

def hdf5_from_disk(path, batch_sizes=8, cache_size=16):
  queue = Queue(maxsize=cache_size)

  worker = threading.Thread(
    target=hdf5_batch_worker,
    kwargs=dict(path=path, out_queue=queue, batch_sizes=batch_sizes)
  )

  worker.daemon = True
  worker.start()

  return queue_to_stream(queue)

def np_batch_worker(path, out_queue, batch_size, mmap_mode='r'):
  mmap = np.load(path, mmap_mode=mmap_mode)

  for indx in inf_random_seq(mmap.shape[0], batch_size=batch_size, allow_smaller=False):
    out_queue.put(mmap[indx], block=True)

def np_from_disk(data_root, batch_sizes=8, cache_size=16, mmap_mode='r'):
  bin_patches = [
    osp.join(data_root, 'bin_%d.npy' % i)
    for i in range(len(os.listdir(data_root)))
  ]

  if type(batch_sizes) in [long, int]:
    batch_sizes = [batch_sizes] * len(bin_patches)

  queues = [ Queue(maxsize=cache_size) for _ in bin_patches ]

  workers = [
    threading.Thread(
      target=np_batch_worker,
      kwargs=dict(path=path, out_queue=queue, batch_size=batch_size, mmap_mode=mmap_mode)
    )

    for path, queue, batch_size in zip(bin_patches, queues, batch_sizes)
  ]

  for worker in workers:
    worker.daemon = True
    worker.start()

  return queues_stream(queues)

def queue_to_stream(queue):
  while True:
    yield queue.get(block=True)

def queues_stream(queues):
  it = itertools.izip(*[
    queue_to_stream(queue) for queue in queues
  ])

  for xs in it:
    yield np.vstack(xs)