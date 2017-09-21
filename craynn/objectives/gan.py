import theano
import theano.tensor as T

__all__ = [
  'energy_based'
]

def log_cross_entropy(scores_real, scores_pseudo):
  log_f = T.nnet.softplus(-scores_real)
  log_1_f = T.nnet.softplus(scores_pseudo)

  loss_real = T.mean(log_f)
  loss_pseudo = T.mean(log_1_f)

  return 0.5 * (loss_real + loss_pseudo), -loss_pseudo


def energy_based(X_original, X_generated, discriminator, margin = 1):
  score_original, = discriminator(X_original)
  score_generated, = discriminator(X_generated)

  zero = T.constant(0.0, dtype='float32')
  margin = T.constant(margin, dtype='float32')

  return T.mean(score_original) + T.mean(T.maximum(zero, margin - score_generated)), T.mean(score_generated)