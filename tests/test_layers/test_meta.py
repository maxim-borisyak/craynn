from craynn.layers import *

def test_meta():
  in1 = InputLayer(shape=(1, ), variable=1, name='input1')
  in2 = InputLayer(shape=(1,), variable=2, name='input2')

  plus1 = custom_layer(lambda a, b: a + b, name='plus1')(in1, in2)

  assert get_output([plus1])[0] == 3
  assert get_output([plus1], substitutes={ in1 : 3 })[0] == 5
  assert get_output([plus1], substitutes={ in2 : 4 })[0] == 5
  assert get_output([plus1], substitutes={ in1 : 10, in2 : 999 })[0] == 1009

  plus2 = custom_layer(lambda a, b: a + b, name='plus2')(in1, plus1)

  assert get_output([plus1, plus2]) == [3, 4]

  assert len(get_layers([plus1, plus2])) == 4
  assert len(get_layers(plus2)) == 4
  assert len(get_layers(plus1)) == 3

  assert get_output_shape([plus1]) == [(1, )]
