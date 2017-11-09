from ..layers import *
from ..subnetworks import mse_energy as energy_factory
from ..subnetworks import achain

from . import *

__all__ = [
  'energy_based_'
]

### Renaming trick to making class constructor seem like partial application.
def energy_based_(inputs):
  """
  Usage:

  ```
  EnergyBased(<inputs definition>)(<input preprocessing operator>, <output preprocessing operator>)(
    <wrapped operator>
  )
  ```
  """
  return lambda input_preprocessing_op=nothing, output_preprocessing_op=nothing: lambda *definition: \
    EnergyBased(inputs, achain(*definition), input_preprocessing_op, output_preprocessing_op)

class EnergyBased(Net):
  def __init__(self, inputs, op, input_preprocessing_op=nothing, output_preprocessing_op=nothing):
    super(EnergyBased, self).__init__(
      energy_factory(
        op,
        input_preprocessing_op=input_preprocessing_op,
        output_preprocessing_op=output_preprocessing_op
      ),
      inputs
    )