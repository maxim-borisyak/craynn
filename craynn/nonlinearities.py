import tensorflow as tf

__all__ = [
  'sigmoid',
  'relu',
  'leaky_relu',
  'softplus',
  'softmax',
  'elu',
  'linear'
]

sigmoid = lambda name='sigmoid': lambda x: tf.nn.sigmoid(x, name=None)

relu = lambda name='ReLU': lambda x: tf.nn.relu(x, name=name)
leaky_relu = lambda leakiness=0.05, name='LeakyReLU': \
  lambda x: tf.nn.leaky_relu(x, alpha=leakiness, name=name)

softplus = lambda name='softplus': lambda x: tf.nn.softplus(x)
softmax = lambda name='softmax': lambda x: tf.nn.softmax(x, name=name)
elu = lambda name='ELU': lambda x: tf.nn.elu(x, name=name)

linear = lambda name='linear': lambda x: x