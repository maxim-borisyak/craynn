import theano.tensor as T

from functools import reduce

from ...objectives import energy_based

__all__ = [
  'GAN',
  'CycleGAN',
  'StageGAN'
]

class GAN(object):
  def __init__(self, loss, discriminator, generator):
    """
    Provides GAN model.
    :param loss: loss functions (scores real, scores pseudo), must return two expressions
      loss for discriminator training and generator loss for **minimization** .
    :param discriminator: a `craynn.Expression` should return collection with one value.
    :param generator: a `craynn.Expression` should return collection with one value.
    """
    self.loss = loss
    self.discriminator = discriminator
    self.generator = generator


  def __call__(self,
               X_real, Z,
               discriminator_reg=None, generator_reg = None):
    """
    Connects GAN, compiles discriminator step
    :param X_real: theano expression that holds real values
    :param Z: input for generator
    :return:
    """

    self.X_pseudo, = self.generator(Z)

    self.scores_real, = self.discriminator(X_real)
    self.scores_pseudo, = self.discriminator(self.X_pseudo)

    self.losses_discriminator, self.losses_generator = self.loss(self.scores_real, self.scores_pseudo)

    self.pure_loss_discriminator = T.mean(self.losses_discriminator)
    self.pure_loss_generator = T.mean(self.losses_generator)

    self.loss_discriminator = self.pure_loss_discriminator
    self.loss_generator = self.pure_loss_generator

    if discriminator_reg is not None:
      self.loss_discriminator += discriminator_reg * self.discriminator.reg_l2()

    if generator_reg is not None:
      self.loss_generator += generator_reg * self.generator.reg_l2()


    return self.loss_discriminator, self.loss_generator, self.pure_loss_discriminator, self.pure_loss_generator


class CycleGAN(object):
  def __init__(self,
               XtoY, discriminator_Y, YtoX=None, discriminator_X=None,
               loss_Y=energy_based(), loss_X=None,
               loss_XY=None, loss_YX=None,
               loss_XYX=None, loss_YXY=None
               ):

    self.XtoY = XtoY
    self.YtoX = YtoX
    self.discriminator_X = discriminator_X
    self.discriminator_Y = discriminator_Y

    self.loss_X = loss_X
    self.loss_Y = loss_Y

    self.loss_XY = loss_XY
    self.loss_YX = loss_YX

    self.loss_XYX = loss_XYX
    self.loss_YXY = loss_YXY

  def __call__(self, X, Y):
    def apply(f, *args):
      if f is None or any([arg is None for arg in args]):
        return None
      else:
        return f(*args)

    ### GAN losses in Y domain
    XY, = self.XtoY(X)

    score_Y, = self.discriminator_Y(Y)
    score_XY, = self.discriminator_Y(XY)

    ### adversarial loss in Y domain, i.e. vanila GAN loss
    self.adv_loss_discriminator_Y, self.adv_loss_XtoY = self.loss_Y(score_Y, score_XY)

    ### losses of X -> Y
    self.semi_cycle_loss_XY = apply(self.loss_XY, X, XY)

    YX, = apply(self.YtoX, Y)
    YXY, = apply(self.XtoY, YX)
    XYX, = apply(self.YtoX, XY)

    ### adversarial loss in X domain
    score_X, = apply(self.discriminator_X, X)
    score_YX, = apply(self.discriminator_X, YX)
    self.adv_loss_discriminator_X, self.adv_loss_YtoX = apply(self.loss_X, score_X, score_YX)

    ### losses of Y -> X
    self.semi_cycle_loss_YX = apply(self.loss_YX, Y, YX)


    ### cycle loss X -> Y -> X
    self.cycle_loss_XYX = apply(self.loss_XYX, X, XYX)

    ### cycle loss Y -> X -> Y
    self.cycle_loss_YXY = apply(self.loss_YXY, Y, YXY)


    self.loss_generator = reduce(lambda a, b: a + b, [
      l for l in [
        self.adv_loss_XtoY, self.adv_loss_YtoX,
        self.semi_cycle_loss_YX, self.semi_cycle_loss_XY,
        self.cycle_loss_YXY, self.cycle_loss_YXY
      ] if l is not None
    ])

    return (
      self.adv_loss_discriminator_Y,
      self.adv_loss_discriminator_X,
      self.loss_generator
    )

class StageGAN(object):
  def __init__(self, loss, discriminators, generator):
    """
    Provides GAN model.
    :param loss: loss functions (scores real, scores pseudo), must return two expressions
      loss for discriminator training and generator loss for **minimization** .
    :param discriminators: a `craynn.Expression` should return **collection** of scores.
    :param generator: a `craynn.Expression` should return collection with one value.
    """
    self.loss = loss
    self.discriminators = discriminators
    self.generator = generator


  def __call__(self, X_real, X_pseudo):
    """
    Connects GAN, compiles discriminator step
    """

    self.scores_real = self.discriminators(X_real)
    self.scores_pseudo = self.discriminators(X_pseudo)

    self.losses_discriminator = []
    self.losses_generator = []

    for sr, sp in zip(self.scores_real, self.scores_pseudo):
      ld, lg = self.loss(sr, sp)
      self.losses_discriminator.append(T.mean(ld))
      self.losses_generator.append(T.mean(lg))

    self.stages_discriminator = []
    self.stages_generator = []

    for ld, lg in zip(self.losses_discriminator, self.losses_generator):
      if len(self.stages_discriminator) == 0:
        self.stages_discriminator.append(ld)
        self.stages_generator.append(lg)
      else:
        ld = self.stages_discriminator[-1] + ld
        lg = self.stages_generator[-1] + lg

        self.stages_discriminator.append(ld)
        self.stages_generator.append(lg)

    for i, sd in enumerate(self.stages_discriminator):
      self.stages_discriminator[i] = (sd / (i + 1)) if i > 0 else sd

    for i, sg in enumerate(self.stages_generator):
      self.stages_generator[i] = (sg / (i + 1)) if i > 0 else sg


    return self.stages_discriminator, self.stages_generator