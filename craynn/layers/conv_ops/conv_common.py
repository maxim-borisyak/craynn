__all__ = [
  'get_stride',
  'get_pad',
  'get_pool_mode',

  'get_kernel_size',
  'get_kernel_output_shape',
  'get_transposed_kernel_output_shape',

  'proper_data_format',
  'strange_data_format'
]

def get_stride(stride, ndim):
  if isinstance(stride, int):
    return (1, ) + (stride, ) * ndim + (1, )

  stride = tuple(stride)

  if len(stride) == ndim:
    ### full stride specified
    return (1, ) + stride + (1, )

  elif len(stride) == ndim + 1:
    ### stride includes channel
    return (1, ) + stride

  elif len(stride) == ndim + 2:
    ### full stride
    if stride[0] > 1:
      import warnings
      warnings.warn('You probably want to reconsider stride=%s' % stride)
    return stride

  else:
    raise ValueError('Stride must be collection of either %d or %d integers or a integer.' % (ndim, ndim - 2))


def get_pad(pad : str):
  assert pad in ('same', 'valid')
  return pad.upper()

def get_pool_mode(pad : str):
  assert pad in ('max', 'mean')

  if pad == 'max':
    return "MAX"
  else:
    return "AVG"

def get_kernel_size(kernel_size, ndim):
  if isinstance(kernel_size, int):
    return (kernel_size,) * ndim
  else:
    assert len(kernel_size) == ndim
    return kernel_size

def get_kernel_output_shape(input_shape, spatial_kernel_size, pad, stride):
  if pad.lower() == 'valid':
    conv_out = lambda n, k, s: (n - k) // s + 1
  else:
    conv_out = lambda n, k, s: (n + s - 1) // s

  ndim = len(spatial_kernel_size)

  return tuple([
    conv_out(n, f, s) for n, f, s in zip(input_shape[-ndim:], spatial_kernel_size[-ndim:], stride[-ndim:])
  ])

def get_transposed_kernel_output_shape(input_shape, spatial_kernel_size, pad, stride):
  if pad.lower() == 'valid':
    deconv_out = lambda n, k, s: n * s + max(k - s, 0)
  else:
    deconv_out = lambda n, k, s: n * s

  ndim = len(spatial_kernel_size)

  return tuple([
    deconv_out(n, f, s) for n, f, s in zip(input_shape[-ndim:], spatial_kernel_size[-ndim:], stride[-ndim:])
  ])

def proper_data_format(ndim):
  if ndim == 1:
    return 'NCW'
  elif ndim == 2:
    return 'NCHW'
  else:
    return 'NCDHW'

def strange_data_format(ndim):
  if ndim == 1:
    return 'NWC'
  elif ndim == 2:
    return 'NHWC'
  else:
    return 'NDHWC'