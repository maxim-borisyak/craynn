import theano
import theano.tensor as T

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
               adversarial_loss_Y=energy_based(), adversarial_loss_X=None,
               loss_XY=None, loss_YX=None,
               loss_XYX=None, loss_YXY=None
               ):

    self.XtoY = XtoY
    self.YtoX = YtoX
    self.discriminator_X = discriminator_X
    self.discriminator_Y = discriminator_Y

    self.adversarial_loss_X = adversarial_loss_X
    self.adversarial_loss_Y = adversarial_loss_Y

    self.loss_XY = loss_XY
    self.loss_YX = loss_YX

    self.loss_XYX = loss_XYX
    self.loss_YXY = loss_YXY

  def __call__(self, X, Y):
    ### GAN losses in Y domain
    XY = self.XtoY(X)

    score_Y, = self.discriminator_Y(Y)
    score_XY, = self.discriminator_Y(XY)

    self.gan_loss_discriminator_Y, self.gan_loss_XtoY = self.adversarial_loss_Y(score_Y, score_XY)

    ### GAN losses in X domain
    YX, = self.YtoX(Y)

    score_X, = self.discriminator_X(X)
    score_YX, = self.discriminator_X(YX)

    self.gan_loss_discriminator_X, self.gan_loss_YtoX = self.adversarial_loss_X(score_X, score_YX)

    ### X -> Y -> X cycle loss

    XYX, = self.YtoX(XY)

    self.cycle_loss_X = self.cycle_loss_X(X, XYX)

    if self.aux_loss_YtoX is not None:
      self.cycle_loss_X += self.cycle_loss_coef_X * self.aux_loss_YtoX(X, XYX)

    ### Y -> X -> Y cycle loss

    Y_cycled, = self.XtoY(YX)
    self.cycle_loss_Y = self.cycle_loss_Y(Y, Y_cycled)

    if self.aux_loss_XtoY is not None:
      self.cycle_loss_Y += self.cycle_loss_coef_Y * self.aux_loss_XtoY(Y, Y_cycled)


    self.full_loss_XtoY = self.gan_loss_XtoY + self.cycle_loss_coef_Y * self.cycle_loss_Y

    self.full_loss_reverse = self.gan_loss_YtoX + self.cycle_loss_coef_X * self.cycle_loss_X


    return (
      self.gan_loss_discriminator_Y,
      self.gan_loss_discriminator_X,
      self.full_loss_XtoY,
      self.full_loss_reverse,
      self.gan_loss_XtoY,
      self.gan_loss_YtoX,
      self.cycle_loss_Y,
      self.cycle_loss_X
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