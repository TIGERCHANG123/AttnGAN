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


class conv(tf.keras.Model):
    def __init__(self, kernel_size, filters, strides, padding, name):
        super(conv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, name=name + '_conv',
                                           padding=padding, kernel_initializer=RandomNormal(stddev=0.02))
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, name=name + '_bn')
        self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2, name=name + '_leakyRelu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyRelu(x)
        return x


class deconv(tf.keras.Model):
    def __init__(self, filters, strides, padding, name):
        super(deconv, self).__init__()
        self.conv = layers.Conv2DTranspose(filters, kernel_size=3,
                                           strides=strides, padding=padding,
                                           use_bias=False, name=name + '_conv',
                                           kernel_initializer=RandomNormal(stddev=0.02))
        self.bn = layers.BatchNormalization(momentum=0.9, name=name + '_bn')
        # self.relu = tf.keras.layers.ReLU(name=name+'_relu')
        self.relu = GLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Resdual_Block(tf.keras.Model):
  def __init__(self, filters, name):
      super(Resdual_Block, self).__init__()
      self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=RandomNormal(stddev=0.02), name=name+'_conv1')
      self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=RandomNormal(stddev=0.02), name=name+'_conv2')
      self.bn1 = tf.keras.layers.BatchNormalization(name=name+'_bn1')
      self.bn2 = tf.keras.layers.BatchNormalization(name=name+'_bn2')
      self.Relu1 = GLU()
      # self.Relu2 = GLU()
  def call(self, x):
      # x -> R(w * h * c)
      y = self.conv1(x)
      # y -> R(w * h * filters)
      y = self.bn1(y)
      y = self.Relu1(y)
      # y -> R(w * h * filters/2)
      y = self.conv2(y)
      # y -> R(w * h * filters/2)
      y = self.bn2(y)
      y = y + x
      return y

class attention(tf.keras.Model):
  def __init__(self, units, name):
    super(attention, self).__init__()
    self.dense = layers.Dense(units=units, name=name+'_dense')
  def call(self, e, h):
    # e -> R(D*T), h -> R(h * w * D^)
    e_ = tf.transpose(e, [0, 2, 1])
    # e_ -> R(T * D)
    e_ = self.dense(e_)
    # e_ -> R(T * D^)
    e_ = tf.transpose(e_, [0, 2, 1])
    # e_ -> R(D^*T)
    h1 = tf.transpose(h, [0, 3, 1, 2])
    # h1 -> R(D^ * h * w)
    h1 = tf.reshape(h1, [h1.shape[0], h1.shape[1], -1])
    # h1 -> R(D^ * N)
    s = tf.matmul(tf.transpose(e_, [0, 2, 1]), h1)
    # s -> R(T * N)
    beta = tf.nn.softmax(s, axis=1)
    c = tf.matmul(e_, beta)
    # c -> R(D^ * N)
    c = tf.transpose(c, [0, 2, 1])
    # c -> R(N * D^)
    c = tf.reshape(c, h.shape)
    # c -> R(h * w * D^)
    output = tf.concat([c, h], axis=3)
    # output -> R(h * w * 2D^)
    return output

class generator_Input(tf.keras.Model):
  def __init__(self, shape, name):
    super(generator_Input, self).__init__()
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

class generator_Output(tf.keras.Model):
  def __init__(self, image_depth, strides, padding, name):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2D(image_depth,kernel_size=3, strides=strides, name=name+'_conv',
                                       padding=padding, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
    self.actv = layers.Activation(activation='tanh', name=name+'_tan')
  def call(self, x):
    x = self.conv(x)
    x = self.actv(x)
    return x

class discriminator_Input(tf.keras.Model):
  def __init__(self, filters, name):
    super(discriminator_Input, self).__init__()
    self.conv = keras.layers.Conv2D(filters, kernel_size=4, strides=2, name=name+'_conv',
                                    padding="same", kernel_initializer=RandomNormal(stddev=0.02))
    self.leakyRelu = keras.layers.LeakyReLU(alpha=0.2, name=name+'_leakyRelu')
    self.middle_list = [
        conv(kernel_size=4, filters=filters*2, strides=2, padding='same', name=name+'middle1'),
        conv(kernel_size=4, filters=filters*4, strides=2, padding='same', name=name+'middle2'),
        conv(kernel_size=4, filters=filters*8, strides=2, padding='same', name=name+'middle3'),
    ]
  def call(self, x):
    # x -> R(w * h * c)
    x = self.conv(x)
    # x -> R(w/2 * h/2 * filters)
    x = self.leakyRelu(x)
    for i in range(len(self.middle_list)):
      x = self.middle_list_0[i](x)
    # x -> R(w/16 * h/16 * filters*8)
    return x

class discriminator_Output(tf.keras.Model):
  def __init__(self, name):
      super(discriminator_Output, self).__init__()

      self.flatten = tf.keras.layers.Flatten(name=name+'_flatten')
      self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                           name=name+'output', kernel_initializer=RandomNormal(stddev=0.02))
  def call(self, x):
      y1 = self.flatten(x)
      y1 = self.dense(y1)
      return y1







