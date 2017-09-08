import pickle

import numpy as np
import theano.tensor as T
from crayimage.runutils import BatchStreams
from lasagne import *
from tqdm import tqdm

filename = __file__

config = {
  'channels': [
    (8, 16, 32, 64),
    (8, 8, 16, 16, 32, 32, 64, 64),
    (16, 32, 64, 32, 16)
  ],
  'block_length': [1, 2],
  'block_depth': [5],
  'nonlinearity': [
    nonlinearities.elu,
    nonlinearities.leaky_rectify,
    nonlinearities.tanh
  ],
  'dropout_p': [None, 0.1],
  'filter_size': [(3, 3)],
  'noise_sigma': [0.0],
  'output_channels': [10],
  'output_nonlinearity': [nonlinearities.linear]
}

opt_config = {
  'loss': [
    ('mse', lambda a, b: T.mean((a - b) ** 2))
  ],
  'c_reg': [None, 1.0e-4, 1.0e-3, 1.0e-2]
}

results = []
format = lambda d: ('{\n  ' + '\n  '.join(['%s = %s' % (k, v) for k, v in d.items()]) + '\n}')
format_pickle = lambda d: dict([ (k, str(v)) for k, v in d.items() ])

def get_seq(d):
  import itertools
  kvs = [[(k, v) for v in vs] for k, vs in d.items()]
  return [dict(kv) for kv in itertools.product(*kvs)]

def softmax2d(x):
  exped = T.exp(x - x.max(axis=1, keepdims=True))
  return exped / exped.sum(axis=1, keepdims=True)

data = np.load('/ssd/scratch/mborisyak/multi_mnist.npz')

images_ = data['images'].reshape(-1, 1, 64, 64)
targets_ = data['targets'].reshape(-1, 10, 64, 64)

data = np.load('/ssd/scratch/mborisyak/multi_mnist_single_test.npz')

images_test = data['images'].reshape(-1, 1, 64, 64)
targets_test = data['targets'].reshape(-1, 10, 64, 64)

X = T.ftensor4()
y = T.ftensor4()

from craynn import DiffusionNet

def evaluate(predict):
  def get_predictions(pred):
    b = np.zeros(shape=(pred.shape[0],), dtype='uint8')

    for i in xrange(pred.shape[0]):
      ### pixel-wise prediction
      ps = np.argmax(pred[i], axis=0).ravel()
      ### image-wise prediction
      bins = np.bincount(ps)
      b[i] = np.uint8(np.argmax(bins))

    return b

  pred = BatchStreams.traverse(predict, images_test)
  p = get_predictions(pred)
  t = np.min(targets_test, axis=(1, 2))
  acc = np.mean(p == t)

  mse = np.mean((pred - targets_test) ** 2)

  return mse, acc


f = open("%s.log.txt" % filename, 'w')

for conf in tqdm(get_seq(config), desc='config cycle'):
  for opt_conf in tqdm(get_seq(opt_config), desc='opt config cycle'):
    _, loss_f = opt_conf['loss']
    c_reg = opt_conf['c_reg']

    diff_net = DiffusionNet(
      img_shape=(1, 64, 64),
      **conf
    )

    predictions, = diff_net(X)
    probs = softmax2d(predictions)

    pure_loss = loss_f(predictions, y)
    mse = T.mean((predictions - y) ** 2)

    reg = 1.0e+3 * diff_net.transfer_reg() + 1.0e-2 * diff_net.redistribution_reg() + 1.0e-3 * diff_net.reg_l2()
    loss = pure_loss if c_reg is None else (pure_loss + c_reg * reg)

    params = layers.get_all_params(diff_net.outputs, trainable=True)

    upd = updates.adamax(loss, params, learning_rate=1.0e-3)
    train_mse = theano.function([X, y], [mse], updates=upd)

    train = train_mse
    #train = lambda *args: [1.0]
    #predict = lambda x: np.random.uniform(size=(x.shape[0], 10)+ x.shape[2:])
    predict = theano.function([X], probs)

    ### first stage

    n_epoches = 64
    n_batches = 128
    mse_losses = np.ndarray(shape=(n_epoches, n_batches), dtype='float32')

    for i in tqdm(range(n_epoches), desc='epoches'):
      stream = BatchStreams.random_batch_stream(images_.shape[0], batch_size=64, n_batches=n_batches)

      for j, ind in enumerate(stream):
        mse_losses[i, j], = train(images_[ind].astype('float32') / 64.0, targets_[ind])

    mse, acc = evaluate(predict)

    f.write("%s %s\nMSE: %.3e, ACC: %.3e\n\n" % (
      format(conf), format(opt_conf), mse, acc
    ))

    f.flush()

    results.append(
      ((format_pickle(conf), format_pickle(opt_conf)), (mse, acc))
    )

    with open('%s.log.pickled' % filename, 'w') as g:
      pickle.dump(results, g)

f.close()