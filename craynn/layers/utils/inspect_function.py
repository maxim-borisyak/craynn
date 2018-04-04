"""
The MIT License (MIT)

Copyright (c) 2014-2015 Lasagne contributors

Lasagne uses a shared copyright model: each contributor holds copyright over
their contributions to Lasagne. The project versioning records all such
contribution and copyright details.
By contributing to the Lasagne repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__all__ = [
  'inspect_kwargs',
  'apply_with_kwrags'
]

def inspect_kwargs(func):
  """
  Inspects a callable and returns a list of all optional keyword arguments.
  Parameters
  ----------
  func : callable
      The callable to inspect
  Returns
  -------
  kwargs : list of str
      Names of all arguments of `func` that have a default value, in order
  """
  # We try the Python 3.x way first, then fall back to the Python 2.x way
  try:
    from inspect import signature
  except ImportError:  # pragma: no cover
    from inspect import getargspec
    spec = getargspec(func)
    return spec.args[-len(spec.defaults):] if spec.defaults else []
  else:  # pragma: no cover
    params = signature(func).parameters
    return [p.name for p in params.values() if p.default is not p.empty]

def apply_with_kwrags(f, *args, **kwargs):
  accepted_kwargs = inspect_kwargs(f)
  passed_kwargs = dict()

  for k, v in kwargs.items():
    if k in accepted_kwargs:
      passed_kwargs[k] = v

  return f(*args, **passed_kwargs)