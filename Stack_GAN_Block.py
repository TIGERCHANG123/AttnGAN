import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import *

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.flatten = layers.Flatten()
        # self.gru = layers.GRU(self.enc_units,
        #                                return_sequences=True,
        #                                return_state=True,
        #                                recurrent_initializer='glorot_uniform')
        self.dense = layers.Dense(latent_dim)
        self.leakyRelu = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        x = self.leakyRelu(x)
        return x

class generator_Input(tf.keras.Model):
  def __init__(self, shape):
    super(generator_Input, self).__init__()
    self.dense = layers.Dense(shape[0] * shape[1] * shape[2], use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
    self.reshape = layers.Reshape([shape[0], shape[1], shape[2]])
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.relu = tf.keras.layers.ReLU()
  def call(self, x):
    x = self.dense(x)
    x = self.reshape(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class Resdual_Block(tf.keras.Model):
  def __init__(self, filters):
      super(Resdual_Block, self).__init__()
      self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=RandomNormal(stddev=0.02))
      self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                                         kernel_initializer=RandomNormal(stddev=0.02))
      self.bn1 = tf.keras.layers.BatchNormalization()
      self.bn2 = tf.keras.layers.BatchNormalization()
      self.Relu1 = tf.keras.layers.ReLU()
      self.Relu2 = tf.keras.layers.ReLU()
  def call(self, x):
      y = self.conv1(x)
      y = self.bn1(y)
      y = self.Relu1(y)
      y = self.conv2(y)
      y = self.bn(y)
      y = y + x
      y = self.Relu2(y)
      return y

class generator_conv(tf.keras.Model):
    def __init__(self, filters, strides, padding):
        super(generator_conv, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size=5, strides=strides, padding=padding, use_bias=False,
                                           kernel_initializer=RandomNormal(stddev=0.02))
        self.bn = layers.BatchNormalization(momentum=0.9)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class generator_deconv(tf.keras.Model):
  def __init__(self, filters, strides, padding):
      super(generator_deconv, self).__init__()
      self.conv = layers.Conv2DTranspose(filters, kernel_size=5, strides=strides, padding=padding, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
      self.bn = layers.BatchNormalization(momentum=0.9)
      self.relu = tf.keras.layers.ReLU()
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class generator_Output(tf.keras.Model):
  def __init__(self, image_depth, strides, padding):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2DTranspose(image_depth, kernel_size=5, strides=strides, padding=padding, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
    self.actv = layers.Activation(activation='tanh')
  def call(self, x):
    x = self.conv(x)
    x = self.actv(x)
    return x

class discriminator_Input(tf.keras.Model):
  def __init__(self, filters, strides):
    super(discriminator_Input, self).__init__()
    self.conv = keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same", kernel_initializer=RandomNormal(stddev=0.02))
    self.leakyRelu = keras.layers.LeakyReLU(alpha=0.2)
    # self.dropout = keras.layers.Dropout(0.3)
  def call(self, x):
    x = self.conv(x)
    x = self.leakyRelu(x)
    # x = self.dropout(x)
    return x

class discriminator_Middle(tf.keras.Model):
  def __init__(self, kernel_size, filters, strides, padding):
      super(discriminator_Middle, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=RandomNormal(stddev=0.02))
      self.bn = tf.keras.layers.BatchNormalization()
      self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
      # self.dropout = tf.keras.layers.Dropout(0.3)
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.leakyRelu(x)
      # x = self.dropout(x)
      return x

class discriminator_Output_channel(tf.keras.Model):
  def __init__(self, filters, strides, with_activation):
      super(discriminator_Output_channel, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same", kernel_initializer=RandomNormal(stddev=0.02))
      self.flatten = tf.keras.layers.Flatten(name='output')
      if with_activation:
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output')
      else:
        self.dense = tf.keras.layers.Dense(1, name='output')
  def call(self, x):
      x = self.conv(x)
      y = self.flatten(x)
      y = self.dense(y)
      return y

class discriminator_Output(tf.keras.Model):
  def __init__(self, with_activation):
      super(discriminator_Output, self).__init__()
      self.flatten = tf.keras.layers.Flatten()
      if with_activation:
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output', kernel_initializer=RandomNormal(stddev=0.02))
      else:
        self.dense = tf.keras.layers.Dense(1, name='output')
  def call(self, x):
      y1 = self.flatten(x)
      y1 = self.dense(y1)
      return y1







