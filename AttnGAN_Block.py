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
    def __init__(self, filters, name):
        super(deconv, self).__init__()
        self.upsample = layers.UpSampling2D(2,  interpolation='nearest')
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, name=name + '_conv',
                                           padding='same', kernel_initializer=RandomNormal(stddev=0.02))
        # self.conv = layers.Conv2DTranspose(filters, kernel_size=3,
        #                                    strides=2, padding='same',
        #                                    use_bias=False, name=name + '_conv',
        #                                    kernel_initializer=RandomNormal(stddev=0.02))
        self.bn = layers.BatchNormalization(momentum=0.9, name=name + '_bn')
        self.relu = GLU()

    def call(self, x):
        x = self.upsample(x)
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
      # y -> R(w * h * filters)
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
    self.bn = layers.BatchNormalization(momentum=0.9, name=name + '_bn')
    self.reshape = layers.Reshape([shape[0], shape[1], shape[2]//2], name=name+'_reshape')
    # self.relu = tf.keras.layers.ReLU()
    # self.relu = GLU()
  def call(self, x):
    x = self.dense(x)
    x = self.bn(x)
    x= x[:, :x.shape[1]//2] * tf.keras.activations.sigmoid(x[:, x.shape[1]//2:])
    x = self.reshape(x)
    return x

class generator_Output(tf.keras.Model):
  def __init__(self, name):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2D(3,kernel_size=3, strides=1, name=name+'_conv',
                                       padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
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
      x = self.middle_list[i](x)
    # x -> R(w/16 * h/16 * filters*8)
    return x

class discriminator_Output(tf.keras.Model):
  def __init__(self, ndf, name):
      super(discriminator_Output, self).__init__()
      self.jointConv = layers.Conv2D(filters=ndf * 8, kernel_size=3, strides=1, name=name+'Jointconv',
                                       padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
      self.conv = layers.Conv2D(1, kernel_size=4, strides=4, name=name + '_conv',
                                padding='valid', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
  def call(self, x, text_embedding):
      # text_embedding -> R(D)
      code = tf.expand_dims(text_embedding, axis=1)
      code = tf.expand_dims(code, axis=2)
      # code -> R(1 * 1 * D)
      ones0 = tf.ones(shape=[1, x.shape[1], x.shape[2], 1], dtype=x.dtype)
      # ones0 -> R(4 * 4 * 1)
      image_text0 = ones0 * code
      # image_text0 -> R(4 * 4 * D)
      x = tf.concat([x, image_text0], axis=-1)
      # x -> R(4 * 4 * (ndf*8+D))
      x = self.jointConv(x)
      # x -> R(4 * 4 * ndf*8)
      y = self.conv(x)
      # y -> R(1 * 1 * 1)
      y = tf.keras.activations.sigmoid(y)
      # print('mean y: ', (tf.reduce_mean(y)).numpy())
      return tf.squeeze(y)







