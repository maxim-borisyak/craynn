from .func_magic import partial_with_defaults

__all__ = [
  'based_on',
  'curry_incoming'
]

class DefaultsFactory(object):
  def __init__(self, base_class, class_name, fixed_arguments):
    self.base_class = base_class
    self.class_name = class_name
    self.fixed = fixed_arguments

  def with_defaults(self, **kwargs):
    return derivative(self.base_class, self.class_name, self.fixed, kwargs)

class FixedFactory(object):
  def __init__(self, base_class, class_name):
    self.base_class = base_class
    self.class_name = class_name

  def let(self, **kwargs):
    return DefaultsFactory(self.base_class, self.class_name, kwargs)

class DeriveFactory(object):
  def __init__(self, base_class):
    self.base_class = base_class

  def derive(self, class_name):
    return FixedFactory(self.base_class, class_name)

based_on = lambda base_class: DeriveFactory(base_class)

def fix_definition_module(obj):
  import inspect
  stack = None

  try:
    stack = inspect.stack()
    ### 0-th frame - current scope
    ### 1-st frame - caller scope (e.g. derivative)
    ### 2-nd frame - current module
    ### 3-rd frame - scope where my_layer = black_magic(...) occurs
    definition_module = inspect.getmodule(stack[3][0])
    obj.__module__ = definition_module.__name__
  except:
    pass
  finally:
    del stack

  return obj

def derivative(base_class, class_name : str, fixed_arguments, defaults):
  clsdict = dict(
    __init__ = partial_with_defaults(base_class.__init__, fixed_arguments, defaults)
  )

  cls = type(class_name, (base_class, ), clsdict)
  cls = fix_definition_module(cls)

  return cls

def curry_incoming(LayerClass, name):
  """
  Just a fancy equivalent  to `lambda <params>: lambda incoming: LayerClass(incoming, <params>)`.
  """
  import inspect

  original_signature = inspect.Signature.from_callable(LayerClass.__init__)
  new_signature = inspect.Signature(
    parameters=[
      original_signature.parameters[param_name]
      for param_name in list(original_signature.parameters.keys())[2:]
    ],
    return_annotation=original_signature.return_annotation
  )

  def carried(*args, **kwargs):
    ba = new_signature.bind(*args, **kwargs)

    def linker(incoming):
      return LayerClass(incoming, **ba.arguments)

    linker.__doc__ = '%s(%s, %s)' % (
      name,
      ', '.join([ str(arg) for arg in args ]),
      ', '.join([ '%s=%s' % (k, v) for k, v in kwargs.items() ])
    )
    linker.__str__ = lambda : linker.__doc__
    linker.__repr__ = lambda: linker.__doc__
    return linker

  carried.__doc__ = LayerClass.__doc__

  carried.__signature__ = new_signature
  carried = fix_definition_module(carried)
  carried.__str__ = lambda : str(new_signature)
  carried.__repr__ = lambda: str(new_signature)

  return carried