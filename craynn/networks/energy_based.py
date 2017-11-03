from ..subnetworks import mse_energy as energy_factory
from ..subnetworks import achain

from . import *

__all__ = [
  'energy_based_'
]

### Renaming trick to making class constructor seem like partial application.
def energy_based_(inputs):
  return lambda *definition: \
    EnergyBased(inputs, *definition)

class EnergyBased(Net):
  def __init__(self, inputs, *definition):
    super(EnergyBased, self).__init__(
      energy_factory(achain(*definition)),
      inputs
    )