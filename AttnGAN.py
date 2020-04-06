from AttnGAN_Block import *

class embedding(tf.keras.Model):
  def __init__(self, num_encoder_tokens, embedding_dim, latent_dim):
    super(embedding, self).__init__()
    self.embedding = layers.Embedding(num_encoder_tokens, embedding_dim)
    self.lstm = layers.LSTM(units=latent_dim, return_sequences=True, return_state=True)
    self.go_backwards_lstm = layers.LSTM(units=latent_dim, return_sequences=True, return_state=True, go_backwards=True)
  def call(self, text):
    # print('text shape', text.shape)
    code = self.embedding(text)
    # print('code shape', code.shape)
    whole_sequence_output_1, final_memory_state_1, final_carry_state_1 = self.lstm(code)
    whole_sequence_output_2, final_memory_state_2, final_carry_state_2 = self.go_backwards_lstm(code)

    whole_sequence_output = tf.concat([whole_sequence_output_1, whole_sequence_output_2], axis=-1)
    whole_sequence_output = tf.transpose(whole_sequence_output, [0, 2, 1])
    final_memory_state = tf.concat([final_memory_state_1, final_memory_state_2], axis=-1)

    return whole_sequence_output, final_memory_state

class generate_condition(tf.keras.Model):
  def __init__(self, units):
    super(generate_condition, self).__init__()
    self.units = units
    self.flatten = tf.keras.layers.Flatten()
    self.Dense = tf.keras.layers.Dense(units)
    self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
  def call(self, x):
    x = self.flatten(x)
    x = self.Dense(x)
    x = self.leakyRelu(x)
    return x[:, :int(self.units/2)], x[:, int(self.units/2):]

class Attn_generator(tf.keras.Model):
  def __init__(self):
    super(Attn_generator, self).__init__()
    self.input_layer = Input(shape=[4, 4, 512], name='h0_input')

    self.deconv_list_1 = [
      deconv(filters=256, strides=2, padding='same', name='h0_deconv1'),#256*8*8
      deconv(filters=128, strides=2, padding='same', name='h0_deconv2'),#128*16*16
      deconv(filters=64, strides=2, padding='same', name='h0_deconv3'),#64*32*32
    ]
    self.attention = attention(64, name='h1_attn')
    self.deconv_list_2 = [
      Resdual_Block(128, name='h1_res1'),
      Resdual_Block(128, name='h1_res2'),
      deconv(filters=64, strides=2, padding='same', name='h1_deconv'),  # 256*16*16
    ]
    self.output_0 = generator_Output(image_depth=3, strides=2, padding='same', name='h0_output')#3*64*64
    self.output_1 = generator_Output(image_depth=3, strides=2, padding='same', name='h1_output')
  def call(self, sequence, text_embedding, noise):
    # print('Stage1 text embedding shape: {}, noise shape{}'.format(text_embedding.shape, noise.shape))
    # print('text embedding shape', text_embedding.shape)
    x = tf.concat([noise, text_embedding], axis=-1)
    # print('concat shape: ', x.shape)
    h0 = self.input_layer(x)
    # print('input shape', h0.shape)
    for i in range(len(self.deconv_list_1)):
      h0 = self.deconv_list_1[i](h0)
      # print('h0 deconv {} shape {}'.format(i, h0.shape))
    h1 = self.attention(sequence, h0)
    for i in range(len(self.deconv_list_2)):
      h1 = self.deconv_list_2[i](h1)
      # print('h1 deconv {} shape {}'.format(i, h1.shape))
    out0 = self.output_0(h0)
    out1 = self.output_1(h1)
    # print('output0 shape', out0.shape)
    # print('output1 shape: ', out1.shape)
    return out0, out1

class Attn_discriminator(tf.keras.Model):
  def __init__(self):
    super(Attn_discriminator, self).__init__()
    self.input_layer_0 = discriminator_Input(filters=64, strides=1, name='h0_input')
    self.middle_list_0 = [
      discriminator_Middle(kernel_size=3, filters=64, strides=2, padding='same', name='h0_middle1'),#1024*4*4
      discriminator_Middle(kernel_size=3, filters=128, strides=2, padding='same', name='h0_middle2'),#512*8*8
      discriminator_Middle(kernel_size=3, filters=256, strides=2, padding='same', name='h0_middle3'),#256*16*16
    ]
    self.output_layer_0 = discriminator_Output(with_activation=False, name='h0_output')#3*64*64

    self.input_layer_1 = discriminator_Input(filters=64, strides=1, name='h1_input')
    self.middle_list_1 = [
      discriminator_Middle(kernel_size=3, filters=64, strides=2, padding='same', name='h1_middle1'),  # 1024*4*4
      discriminator_Middle(kernel_size=3, filters=128, strides=2, padding='same', name='h1_middle2'),  # 512*8*8
      discriminator_Middle(kernel_size=3, filters=256, strides=2, padding='same', name='h1_middle3'),  # 256*16*16
    ]
    self.output_layer_1 = discriminator_Output(with_activation=False, name='h1_output')  # 3*64*64
  def call(self, text_embedding, image_0, image_1):
    code = text_embedding
    code = tf.expand_dims(code, axis=1)
    code = tf.expand_dims(code, axis=2)
    # print('code shape: ', code.shape)
    x0 = self.input_layer_0(image_0)
    # print('x0 shape', x0.shape)
    for i in range(len(self.middle_list_0)):
      x0 = self.middle_list_0[i](x0)
      # print('middle {} shape'.format(i), x0.shape)
    ones0 = tf.ones(shape=[1, x0.shape[1], x0.shape[2], 1], dtype=x0.dtype)
    image_text0 = ones0 * code
    x0 = tf.concat([x0, image_text0], axis=-1)
    output0 = self.output_layer_0(x0)
    # print('disc output0 shape: ', output0.shape)
    x1 = self.input_layer_1(image_1)
    # print('x1 shape', x1.shape)
    for i in range(len(self.middle_list_1)):
      x1 = self.middle_list_1[i](x1)
      # print('middle {} shape'.format(i), x1.shape)
    ones1 = tf.ones(shape=[1, x1.shape[1], x1.shape[2], 1], dtype=x1.dtype)
    image_text1 = ones1 * code
    x1 = tf.concat([x1, image_text1], axis=-1)
    output1 = self.output_layer_1(x1)
    # print('disc output1 shape: ', output1.shape)
    return output0, output1

def get_gan(num_tokens):
  Dense_mu_sigma = generate_condition(256*2)
  Embedding = embedding(num_encoder_tokens=num_tokens, embedding_dim=256, latent_dim=128)
  Generator = Attn_generator()
  Discriminator = Attn_discriminator()
  gen_name = 'AttnGAN_2'
  return Dense_mu_sigma, Embedding, Generator, Discriminator, gen_name


