import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import *

class GLU(tf.keras.Model):
    def __init__(self):
        super(GLU, self).__init__()
    def call(self, x):
        nc = x.shape[3]
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :, :, :nc] * tf.keras.activations.sigmoid(x[:, :, :, nc:])

class Input(tf.keras.Model):
  def __init__(self, shape, name):
    super(Input, self).__init__()
    self.dense = layers.Dense(shape[0] * shape[1] * shape[2], use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), name=name+'_dense')
    self.reshape = layers.Reshape([shape[0], shape[1], shape[2]], name=name+'_reshape')
    self.bn = layers.BatchNormalization(momentum=0.9, name=name+'_bn')
    # self.relu = tf.keras.layers.ReLU()
    self.relu = GLU()
  def call(self, x):
    x = self.dense(x)
    x = self.reshape(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class attention(tf.keras.Model):
  def __init__(self, units, name):
    super(attention, self).__init__()
    self.dense = layers.Dense(units=units, name=name+'_dense')
  def call(self, e, h):# e -> R(D*T), h -> R(h * w * D^)
    # e_ -> R(T * D)
    e_ = tf.transpose(e, [0, 2, 1])
    # print('e_ shape', e_.shape)
    # e_ -> R(T * D^)
    e_ = self.dense(e_)
    # print('e_ shape', e_.shape)
    # e_ -> R(D^*T)
    e_ = tf.transpose(e_, [0, 2, 1])
    # print('e_ shape', e_.shape)
    # h1 -> R(D^ * h * w)
    h1 = tf.transpose(h, [0, 3, 1, 2])
    # h1 -> R(D^ * N)
    h1 = tf.reshape(h1, [h1.shape[0], h1.shape[1], -1])
    # print('h1 shape', h1.shape)
    # s -> R(T * N)
    s = tf.matmul(tf.transpose(e_, [0, 2, 1]), h1)
    beta = tf.nn.softmax(s, axis=1)
    # c -> R(D^ * N)
    c = tf.matmul(e_, beta)
    # c -> R(N * D^)
    c = tf.transpose(c, [0, 2, 1])
    # c -> R(h * w * D^)
    c = tf.reshape(c, h.shape)
    # output -> R(h * w * 2D^)
    output = tf.concat([c, h], axis=3)
    return output

class Resdual_Block(tf.keras.Model):
  def __init__(self, filters, name):
      super(Resdual_Block, self).__init__()
      self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=RandomNormal(stddev=0.02), name=name+'_conv1')
      self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=RandomNormal(stddev=0.02), name=name+'_conv2')
      self.bn1 = tf.keras.layers.BatchNormalization(name=name+'_bn1')
      self.bn2 = tf.keras.layers.BatchNormalization(name=name+'_bn2')
      # self.Relu1 = tf.keras.layers.ReLU(name=name+'_relu1')
      # self.Relu2 = tf.keras.layers.ReLU(name=name+'_relu2')
      self.Relu1 = GLU()
      # self.Relu2 = GLU()
  def call(self, x):
      y = self.conv1(x)
      y = self.bn1(y)
      y = self.Relu1(y)
      y = self.conv2(y)
      y = self.bn2(y)
      y = y + x
      # y = self.Relu2(y)
      return y


class deconv(tf.keras.Model):
  def __init__(self, filters, strides, padding, name):
      super(deconv, self).__init__()
      self.conv = layers.Conv2DTranspose(filters, kernel_size=3,
                                         strides=strides, padding=padding,
                                         use_bias=False, name=name+'_conv',
                                         kernel_initializer=RandomNormal(stddev=0.02))
      self.bn = layers.BatchNormalization(momentum=0.9, name=name+'_bn')
      # self.relu = tf.keras.layers.ReLU(name=name+'_relu')
      self.relu = GLU()
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class generator_Output(tf.keras.Model):
  def __init__(self, image_depth, strides, padding, name):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2DTranspose(image_depth,
                                       kernel_size=3, strides=strides, name=name+'_conv',
                                       padding=padding, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
    self.actv = layers.Activation(activation='tanh', name=name+'_tan')
  def call(self, x):
    x = self.conv(x)
    x = self.actv(x)
    return x

class discriminator_Input(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides, name):
    super(discriminator_Input, self).__init__()
    self.conv = keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, name=name+'_conv',
                                    padding="same", kernel_initializer=RandomNormal(stddev=0.02))
    self.leakyRelu = keras.layers.LeakyReLU(alpha=0.2, name=name+'_leakyRelu')
    # self.dropout = keras.layers.Dropout(0.3)
  def call(self, x):
    x = self.conv(x)
    x = self.leakyRelu(x)
    # x = self.dropout(x)
    return x

class discriminator_Middle(tf.keras.Model):
  def __init__(self, kernel_size, filters, strides, padding, name):
      super(discriminator_Middle, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, name=name+'_conv',
                                         padding=padding, kernel_initializer=RandomNormal(stddev=0.02))
      self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, name=name+'_bn')
      self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2, name=name+'_leakyRelu')
      # self.dropout = tf.keras.layers.Dropout(0.3)
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.leakyRelu(x)
      # x = self.dropout(x)
      return x

class discriminator_Output(tf.keras.Model):
  def __init__(self, with_activation, name):
      super(discriminator_Output, self).__init__()
      self.flatten = tf.keras.layers.Flatten(name=name+'_flatten')
      if with_activation:
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                           name=name+'output', kernel_initializer=RandomNormal(stddev=0.02))
      else:
        self.dense = tf.keras.layers.Dense(1, name=name+'output')
  def call(self, x):
      y1 = self.flatten(x)
      y1 = self.dense(y1)
      return y1







