import os
import os.path as osp

import threading
try:
  from queue import Queue
except ImportError as e:
  from Queue import Queue


import numpy as np

from .simple import *

__all__ = [
  'hdf5_from_disk',
  'hdf5',
  'np_from_disk'
]

try:
  import h5py
except ImportError:
  h5py = None


def hdf5_batch_worker(path, out_queue, batch_sizes):
  if h5py is None:
    raise ImportError('Please, make sure h5py package is installed')

  f = h5py.File(path, mode='r')

  n_bins = len([ k for k in  f.keys() if k.startswith('bin_') ])

  datasets = [
    f['bin_%d' % i] for i in range(n_bins)
  ]

  if type(batch_sizes) is int:
    batch_sizes = [batch_sizes] * len(datasets)

  indxes_stream = zip(*[
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
  if h5py is None:
    raise ImportError('Please, make sure h5py package is installed')

  f = h5py.File(path, mode='r')

  n_bins = len([ k for k in  f.keys() if k.startswith('bin_') ])

  datasets = [
    f['bin_%d' % i] for i in range(n_bins)
  ]

  if type(batch_sizes) is int:
    batch_sizes = [batch_sizes] * len(datasets)

  indxes_stream = zip(*[
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

  if type(batch_sizes) is int:
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
  it = zip(*[
    queue_to_stream(queue) for queue in queues
  ])

  for xs in it:
    yield np.vstack(xs)
