import theano.tensor as T
from lasagne.objectives import binary_accuracy, binary_crossentropy, binary_hinge_loss
from lasagne.objectives import categorical_accuracy, categorical_crossentropy

__all__ = [
  'binary_cross_entropy_logit',
  'binary_accuracy', 'binary_crossentropy', 'binary_hinge_loss',
  'categorical_accuracy', 'categorical_crossentropy'
]

def binary_cross_entropy_logit(logit, y):
  return y * T.nnet.softplus(-logit) + (1 - y) * T.nnet.softplus(logit)