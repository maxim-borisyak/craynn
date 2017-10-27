from lasagne import layers

from ..utils import border_mask

__all__ = [
  'MaskLayer', 'mask'
]

class MaskLayer(layers.Layer):
  def __init__(self, incoming, exclude=0, name=None):
    self.exclude = exclude
    self.incoming = incoming
    if exclude > 0:
      self.mask = border_mask(exclude, layers.get_output_shape(incoming))
    else:
      self.mask = None
    
    super(MaskLayer, self).__init__(incoming, name)

  def get_output_for(self, input, **kwargs):
    if self.mask is not None:
      return input * self.mask
    else:
      return input

  def get_output_shape_for(self, input_shape):
    return input_shape


mask = lambda exclude=0: lambda incoming: MaskLayer(incoming, exclude)