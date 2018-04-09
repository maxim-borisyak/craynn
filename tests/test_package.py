import importlib.util as imp

import os
import os.path as osp

def get_submodules(module):
  pathes = [
    item
    for item in os.listdir(osp.dirname(module.__file__))
    if (not item.startswith('_')) and item.endswith('.py')
  ]

  submodules = []

  for path in pathes:
    submodule_name = '%s.%s' % (module.__name__, path.split('.')[0])
    submodule_path = osp.join(osp.dirname(module.__file__), path)
    spec = imp.spec_from_file_location(submodule_name, submodule_path)

    if spec is None:
      raise ValueError('Can not load %s [%s]' % (submodule_name, submodule_path))

    submodule = imp.module_from_spec(spec)
    if submodule is None:
      raise ValueError('Can not load %s [%s]' % (submodule_name, submodule_path))

    spec.loader.exec_module(submodule)
    submodules.append(submodule)

  return submodules


get_exported_names = lambda module: [
  s for s in getattr(module, '__all__', []) if not s.startswith('_')
]

get_exported = lambda module: [
  getattr(module, s) for s in dir(module) if not s.startswith('_')
]

def uniqness(module):
  submodules = get_submodules(module)

  mapping = dict()

  for submodule in submodules:
    for symbol in get_exported_names(submodule):
      if symbol in mapping:
        mapping[symbol].append(submodule)
      else:
        mapping[symbol] = [submodule]

  for symbol in mapping:
    if len(mapping[symbol]) > 1:
      raise Exception(
        'Collision for exported symbol %s, defined in:\n%s' % (
          symbol, '\n'.join( '  %d. %s' % (i, m) for i, m in enumerate(mapping[symbol]))
        )
      )


def test_name_uniqness():
  import craynn
  submodules = [
    getattr(craynn, item)
    for item in [
      'layers'
    ]
  ]

  for submodule in submodules:
    uniqness(submodule)


