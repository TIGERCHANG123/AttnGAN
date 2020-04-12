from AttnGAN_Block import *

class embedding(tf.keras.Model):
  def __init__(self, num_encoder_tokens, embedding_dim, latent_dim):
    super(embedding, self).__init__()
    self.embedding = layers.Embedding(num_encoder_tokens, embedding_dim)
    self.lstm = layers.LSTM(units=latent_dim, return_sequences=True, return_state=True)
    self.go_backwards_lstm = layers.LSTM(units=latent_dim, return_sequences=True, return_state=True, go_backwards=True)
  def call(self, text):
    # text -> R(T)
    # code -> R(T * D_)
    code = self.embedding(text)
    whole_sequence_output_1, final_memory_state_1, final_carry_state_1 = self.lstm(code)
    whole_sequence_output_2, final_memory_state_2, final_carry_state_2 = self.go_backwards_lstm(code)
    # whole_sequence_output1,2 -> R(T * D), final_memory_state1,2 -> R(D)
    whole_sequence_output = tf.concat([whole_sequence_output_1, whole_sequence_output_2], axis=-1)
    # whole_sequence_output -> R(T * 2D)
    whole_sequence_output = tf.transpose(whole_sequence_output, [0, 2, 1])
    # whole_sequence_output -> R(2D * T)
    final_memory_state = tf.concat([final_memory_state_1, final_memory_state_2], axis=-1)
    # final_memory_state -> R(2D)
    return whole_sequence_output, final_memory_state
    # batch_size * (word_length * 2) * seq_length, batch size * (word length * 2)

class generate_condition(tf.keras.Model):
  def __init__(self, units):
    super(generate_condition, self).__init__()
    self.units = units
    self.flatten = tf.keras.layers.Flatten()
    self.Dense = tf.keras.layers.Dense(units * 2)
    # self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
  def call(self, x):
    x = self.flatten(x)
    x = self.Dense(x)
    # x = self.leakyRelu(x)
    x = x[:, :self.units] * tf.keras.activations.sigmoid(x[:, self.units:])
    return x[:, :int(self.units/2)], x[:, int(self.units/2):]

class Attn_generator(tf.keras.Model):
  def __init__(self):
    super(Attn_generator, self).__init__()
    self.input_layer = generator_Input(shape=[4, 4, 1024], name='h0_input')

    self.deconv_list_1 = [
      deconv(filters=512, strides=2, padding='same', name='h0_deconv1'),  # 256*8*8
      deconv(filters=256, strides=2, padding='same', name='h0_deconv1'),#256*8*8
      deconv(filters=128, strides=2, padding='same', name='h0_deconv2'),#128*16*16
      deconv(filters=64, strides=2, padding='same', name='h0_deconv3'),#64*32*32
    ]
    self.attention = attention(32, name='h1_attn')
    self.deconv_list_2 = [
      Resdual_Block(64, name='h1_res1'),
      Resdual_Block(64, name='h1_res2'),
      deconv(filters=64, strides=2, padding='same', name='h1_deconv'),
    ]
    self.output_0 = generator_Output(image_depth=3, strides=1, padding='same', name='h0_output')#3*64*64
    self.output_1 = generator_Output(image_depth=3, strides=1, padding='same', name='h1_output')
  def call(self, sequence, text_embedding, noise):
    # text_embedding -> R(D)
    # noise -> R(100)
    x = tf.concat([noise, text_embedding], axis=1)
    # x -> R((D + 100))
    h0 = self.input_layer(x)
    # h0 -> R(4 * 4 * 1024)
    for i in range(len(self.deconv_list_1)):
      h0 = self.deconv_list_1[i](h0)
    # h0 -> R(64 * 64 * 32)
    h1 = self.attention(sequence, h0)
    # h1 -> R(64 * 64 * 64)
    for i in range(len(self.deconv_list_2)):
      h1 = self.deconv_list_2[i](h1)
    # h1 -> R(128 *128 * 32)
    out0 = self.output_0(h0)
    # out0 -> R(64 * 64 * 3)
    out1 = self.output_1(h1)
    # out1 -> R(128 * 128 * 3)
    return out0, out1

class Attn_discriminator(tf.keras.Model):
  def __init__(self):
    super(Attn_discriminator, self).__init__()

    self.input_layer_0 = discriminator_Input(filters=32, name='h0_input')
    self.conv_0 = layers.Conv2D(filters=32*8,kernel_size=3, strides=1, name='h0_conv',
                                       padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
    self.output_layer_0 = discriminator_Output(name='h0_output')#3*64*64

    self.input_layer_1 = discriminator_Input(filters=32, name='h1_input')
    self.middle_1 = conv(kernel_size=4, filters=32*16, strides=2, padding='same', name='h1_middle')
    self.conv_1 = layers.Conv2D(filters=32*8,kernel_size=3, strides=1, name='h1_output_conv',
                                       padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))
    self.output_layer_1 = discriminator_Output(name='h1_output')  # 3*64*64
  def call(self, text_embedding, image_0, image_1):
    # image_0 -> R(64 * 64 * 3)
    x0 = self.input_layer_0(image_0)
    # x0 -> R(4 * 4 * 256)
    ones0 = tf.ones(shape=[1, x0.shape[1], x0.shape[2], 1], dtype=x0.dtype)
    # text_embedding -> R(D)
    code = text_embedding
    code = tf.expand_dims(code, axis=1)
    code = tf.expand_dims(code, axis=2)
    # code -> R(1 * 1 * D)
    image_text0 = ones0 * code
    # image_text0 -> R(4 * 4 * D)
    x0 = tf.concat([x0, image_text0], axis=-1)
    # x0 -> R(4 * 4 * (256+D))
    x0 = self.conv_0(x0)
    # x0 -> R(4 * 4 * (256+D))
    output0 = self.output_layer_0(x0)

    # image_1 -> R(128 * 128 * 3)
    x1 = self.input_layer_1(image_1)
    # x1 -> R(8 * 8 * 256)
    x1 = self.middle_1(x1)
    # x1 -> (4 * 4 * 1024)
    ones1 = tf.ones(shape=[1, x1.shape[1], x1.shape[2], 1], dtype=x1.dtype)
    image_text1 = ones1 * code
    x1 = tf.concat([x1, image_text1], axis=-1)
    # x0 -> R(4 * 4 * (256+D))
    x1 = self.conv_1(x1)
    output1 = self.output_layer_1(x1)
    return output0, output1

def get_gan(num_tokens):
  Dense_mu_sigma = generate_condition(256*2)
  Embedding = embedding(num_encoder_tokens=num_tokens, embedding_dim=256, latent_dim=128)
  Generator = Attn_generator()
  Discriminator = Attn_discriminator()
  # genenrator的激活函数中使用GLU，res block中最终输出不使用激活函数，各个genenrator的loss加在一起统一后向传播。
  gen_name = 'AttnGAN'
  return Dense_mu_sigma, Embedding, Generator, Discriminator, gen_name


