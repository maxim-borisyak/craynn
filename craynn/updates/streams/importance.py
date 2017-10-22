import numpy as np

__all__ = [
  'uniform_wrt',
  'reweighting'
]

def uniform_bins(target_statistics, sample_per_bin, n_bins=64):
  hist, bins = np.histogram(target_statistics, bins=n_bins)
  indx = np.argsort(target_statistics)
  indicies_categories = np.array_split(indx, np.cumsum(hist)[:-1])
  n_samples = target_statistics.shape[0]

  per_category = sample_per_bin

  weight_correction = (n_bins * np.float64(hist) / n_samples).astype('float32')
  wc = np.repeat(weight_correction, per_category)

  return indicies_categories, wc

def category_sampling(categories_indices, sample_per_bin, n_batches, weights):
  for i in range(n_batches):
    sample = [
      np.random.choice(ind, size=sample_per_bin, replace=True)
      for ind in categories_indices
    ]

    yield np.hstack(sample), weights

def uniform_wrt(target_statistics, sample_per_bin, n_batches, n_bins=64):
  """
  Samples approximately uniformly w.r.t. to `target_statistics`.

  The method computes histogram with uniform binning then samples uniformly from each bin.

  :param target_statistics: values of target statistics for each sample.
  :param sample_per_bin: number of examples to sample from each bin.
  :param n_batches: number of batches to produce.
  :param n_bins: number of bins used for the histogram.
  :return: generator that yields (indices, weights)
  """

  categories_indices, weights = uniform_bins(
    target_statistics=target_statistics,
    sample_per_bin=sample_per_bin,
    n_bins=n_bins
  )

  return category_sampling(categories_indices, sample_per_bin, n_batches, n_bins)


def reweighting(proba, sample_per_bin, n_batches, n_bins=64):
  total = np.sum(proba)

  proba_per_category = total / n_bins
  eps = np.min(proba) / proba_per_category * 1.0e-1

  indx = np.argsort(proba)
  cum_proba = np.cumsum(proba[indx])

  bin_indx = np.floor(cum_proba / proba_per_category - eps)

  split_indx = np.where(
    bin_indx[:-1] != bin_indx[1:]
  )[0] + 1

  categories_indices = np.array_split(indx, split_indx)

  return category_sampling(categories_indices, sample_per_bin, n_batches, n_bins)



