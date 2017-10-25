from lasagne.layers import InputLayer

class AutoInputLayer(InputLayer):
  linked_layer = None

  def get_autoinput(self, linked_input):
    raise NotImplementedError()
