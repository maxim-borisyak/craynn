import tensorflow as tf

def get_grads(loss_or_grads, params):
  assert all([isinstance(param, tf.Variable) for param in params]), 'optimizing a non-free parameter?'

  if isinstance(loss_or_grads, tf.Tensor):
    assert loss_or_grads.shape.ndims < 2 or loss_or_grads.shape.ndims is None, (
      'get %d dimensional cost, should be scalar or vector?' % loss_or_grads.shape.ndims
    )

    return tf.gradients(loss_or_grads, params)
  else:
    assert len(loss_or_grads) == len(params), 'self explanatory'

    return loss_or_grads

