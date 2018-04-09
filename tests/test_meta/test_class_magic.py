from craynn.meta import *

def test_magic():
  class A(object):
    def __init__(self, a, b, c):
      self.a = a
      self.b = b
      self.c = c

    def params(self):
      return (self.a, self.b, self.c)

  B = based_on(A).derive('B').let(a=2).with_defaults(b=1)

  b = B(c=4)
  assert b.params() == (2, 1, 4)

  b = B(4)
  assert b.params() == (2, 1, 4)

  b = B(4, 10)
  assert b.params() == (2, 10, 4)

  b = B(b=4, c=10)
  assert b.params() == (2, 4, 10)

  b = B(c=10, b=4)
  assert b.params() == (2, 4, 10)

  assert b.__module__ == __name__
