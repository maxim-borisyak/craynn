import theano.tensor as T

__all__ = [
  'cross_entropy',
  'energy_based'
]

def _cross_entropy(scores_real, scores_generated, mode=None):
  if mode == 'normal' or mode is None:
    log_f = -T.log(scores_real)
    log_1_f = -T.log(1 - scores_generated)
  elif mode == 'linear':
    log_f = T.nnet.softplus(-scores_real)
    log_1_f = T.nnet.softplus(scores_generated)
  else:
    raise ValueError('Mode should be either normal or linear')


  loss_real = T.mean(log_f)
  loss_pseudo = T.mean(log_1_f)

  return 0.5 * (loss_real + loss_pseudo), loss_pseudo

cross_entropy = lambda mode=None: lambda scores_real, scores_pseudo: \
  _cross_entropy(scores_real, scores_pseudo, mode)

def _energy_based(scores_real, scores_generated, margin = 1):
  zero = T.constant(0.0, dtype='float32')
  loss_discriminator = T.mean(scores_real) + T.mean(T.maximum(zero, margin - scores_generated))
  loss_generator = T.mean(scores_generated)
  return  loss_discriminator, loss_generator

energy_based = lambda margin=1: lambda scores_real, scores_generated: \
  _energy_based(scores_real, scores_generated, margin)