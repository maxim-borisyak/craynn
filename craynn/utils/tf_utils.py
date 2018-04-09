__all__ = [
  'get_dummy_value',
  'dummy_run'
]

def get_dummy_value(X):
  import numpy as np
  shape = tuple(X.shape.as_list())

  shape = tuple([
    shape[i] if shape[i] is not None else 1
    for i in range(len(shape))
  ])

  return np.random.uniform(0.5, 0.55, size=shape).astype(X.dtype.as_numpy_dtype)


def dummy_run(session, request, *placeholders):
  import tensorflow as tf

  session.run(tf.global_variables_initializer())
  return session.run(request, dict([
    (var, get_dummy_value(var))
    for var in placeholders
  ]))