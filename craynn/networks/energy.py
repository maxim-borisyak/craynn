from craynn import Expression
from craynn import plain_mse

from ..layers import Energy2DLayer

__all__ = [
  'EnergyBased'
]

class EnergyBased(Expression):
  def __init__(self, img2img,
               mse=plain_mse,
               input_layer = None):
    if input_layer is None:
      assert len(img2img.inputs) == 1
      self.input_layer = img2img.inputs[0]
    else:
      self.input_layer = input_layer

    outputs = [
        Energy2DLayer(
          [self.input_layer, net],
          energy_function=mse,
          name='MSE'
        )
      for net in img2img.outputs
    ]

    super(EnergyBased, self).__init__(img2img.inputs, outputs)