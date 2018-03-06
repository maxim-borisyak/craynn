def test_mnist(tmpdir):
  from craynn.utils import mnist

  X_train, y_train, X_test, y_test = mnist(tmpdir)

  assert X_train.shape[1:] == (1, 28, 28)
  assert X_test.shape[1:] == (1, 28, 28)

  assert y_test.ndim == 1
  assert y_train.ndim == 1

  X_train, y_train, X_test, y_test = mnist(tmpdir, cast='float32')

  assert X_train.shape[1:] == (1, 28, 28)
  assert X_test.shape[1:] == (1, 28, 28)
  assert X_train.dtype == 'float32'
  assert X_test.dtype == 'float32'

  assert y_test.ndim == 1
  assert y_train.ndim == 1

  X_train, y_train, X_test, y_test = mnist(tmpdir, cast='float32', one_hot=True)

  assert X_train.shape[1:] == (1, 28, 28)
  assert X_test.shape[1:] == (1, 28, 28)
  assert X_train.dtype == 'float32'
  assert X_test.dtype == 'float32'

  assert y_test.shape[1] == 10
  assert y_train.shape[1] == 10