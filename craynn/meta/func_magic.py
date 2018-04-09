import inspect
from collections import OrderedDict

__all__ = [
  'apply_with_kwrags',
  'partial_with_defaults'
]

def partial_with_defaults(f, fixed_arguments, defaults):
  import inspect
  original_signature = inspect.Signature.from_callable(f)

  for fixed in fixed_arguments.keys():
    if fixed not in original_signature.parameters:
      raise TypeError("Trying to fix non-existent parameter %s" % fixed)

  for default in defaults.keys():
    if default not in original_signature.parameters:
      raise TypeError("Trying to set default value for non-existent parameter %s" % default)

  apply_defaults = lambda param: (
    param
    if param.name not in defaults else
    param.replace(default=defaults[param.name])
  )

  ### reorder arguments
  strict_parameters = [
    param_name
    for param_name in original_signature.parameters
    if (
      original_signature.parameters[param_name].default is inspect.Parameter.empty
      and
      param_name not in defaults
    )
  ]

  parameters_with_defaults = [
    param_name
    for param_name in original_signature.parameters
    if (
      original_signature.parameters[param_name].default is not inspect.Parameter.empty
      or
      param_name in defaults
    )
  ]

  new_signature = inspect.Signature(
    parameters=[
      apply_defaults(original_signature.parameters[param_name])
      for param_name in (strict_parameters + parameters_with_defaults)
      if param_name not in fixed_arguments
    ],
    return_annotation=original_signature.return_annotation
  )

  def wrapper(*args, **kwargs):
    ba = new_signature.bind(*args, **kwargs)
    ba.apply_defaults()
    new_kwargs = ba.arguments.copy()
    new_kwargs.update(fixed_arguments)

    return f(**new_kwargs)

  wrapper.__signature__ = new_signature
  wrapper.__doc__ = f.__doc__

  return wrapper

def apply_with_kwrags(f, *args, **kwargs):
  signature = inspect.Signature.from_callable(f)

  if any([ param.kind == inspect.Parameter.VAR_KEYWORD for param in signature ]):
    return f(*args, **kwargs)
  else:
    accepted_kwargs = set([ param.name for param in signature ])
    passed_kwargs = dict()

  for k, v in kwargs.items():
    if k in accepted_kwargs:
      passed_kwargs[k] = v

  return f(*args, **passed_kwargs)