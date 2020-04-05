from AttnGAN_Block import *

class embedding(tf.keras.Model):
  def __init__(self, num_encoder_tokens, embedding_dim, latent_dim):
    super(embedding, self).__init__()
    self.embedding = layers.Embedding(num_encoder_tokens, embedding_dim)
    self.lstm = layers.LSTM(units=embedding_dim, return_sequences=True, return_state=True)
  def call(self, text):
    # print('text shape', text.shape)
    code = self.embedding(text)
    # print('code shape', code.shape)
    whole_sequence_output, final_memory_state, final_carry_state = self.lstm(code)
    return final_memory_state

class Stage1_generator(tf.keras.Model):
  def __init__(self):
    super(Stage1_generator, self).__init__()
    self.input_layer = generator_Input(shape=[4, 4, 512])

    self.middle_layer_list = [
      generator_deconv(filters=256, strides=2, padding='same'),#1024*4*4
      generator_deconv(filters=128, strides=2, padding='same'),#512*8*8
      generator_deconv(filters=64, strides=2, padding='same'),#256*16*16
    ]

    self.output_layer = generator_Output(image_depth=3, strides=2, padding='same')#3*64*64
  def call(self, text_embedding, noise):
    # print('Stage1 text embedding shape: {}, noise shape{}'.format(text_embedding.shape, noise.shape))
    # print('text embedding shape', text_embedding.shape)
    x = tf.concat([noise, text_embedding], axis=-1)
    # print('concat shape: ', x.shape)
    x = self.input_layer(x)
    # print('input shape', x.shape)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
      # print('deconve {} shape {}'.format(i, x.shape))
    x = self.output_layer(x)
    # print('output shape', x.shape)
    return x

class Stage1_discriminator(tf.keras.Model):
  def __init__(self):
    super(Stage1_discriminator, self).__init__()
    self.input_layer = discriminator_Input(filters=64, strides=1)
    self.middle_layer_list1 = [
      discriminator_Middle(kernel_size=5, filters=128, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=256, strides=2, padding='valid'),
    ]
    self.middle_layer_list2 = [
      discriminator_Middle(kernel_size=1, filters=256, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=512, strides=2, padding='valid'),
    ]
    self.output_layer = discriminator_Output(with_activation=False)

  def call(self, text_embedding, x):
    code = text_embedding
    code = tf.expand_dims(code, axis=1)
    code = tf.expand_dims(code, axis=2)

    x = self.input_layer(x)
    for i in range(len(self.middle_layer_list1)):
      x = self.middle_layer_list1[i](x)
    ones = tf.ones(shape=[1, x.shape[1], x.shape[2], 1], dtype=x.dtype)
    image_text = ones * code
    x = tf.concat([x, image_text], axis=-1)
    for i in range(len(self.middle_layer_list2)):
      x = self.middle_layer_list2[i](x)
    # print('Stage1 discriminator x shape: {}'.format(x.shape))
    x = self.output_layer(x)
    return x

class Stage2_generator(tf.keras.Model):
  def __init__(self):
    super(Stage2_generator, self).__init__()

    self.conv_list = [
      generator_conv(filters=64, strides=2, padding='same'),
      generator_conv(filters=128, strides=2, padding='same'),
    ]
    self.res_list = [
      Resdual_Block(256),
      Resdual_Block(256),
      Resdual_Block(256),
      Resdual_Block(256),
    ]

    self.deconv_list = [
      generator_deconv(filters=128, strides=2, padding='same'),#1024*4*4
      generator_deconv(filters=64, strides=2, padding='same'),#512*8*8
    ]

    self.output_layer = generator_Output(image_depth=3, strides=2, padding='same')#3*32*32
  def call(self, text_embedding, x):
    # print('text shape: {}'.format(text.shape))
    # print('x shape: {}'.format(x.shape))
    for i in range(len(self.conv_list)):
          x = self.conv_list[i](x)

    code = text_embedding
    code = tf.expand_dims(code, axis=1)
    code = tf.expand_dims(code, axis=2)
    ones = tf.ones(shape=[1, x.shape[1], x.shape[2], 1], dtype=x.dtype)
    image_text = ones * code
    x = tf.concat([x, image_text], axis=-1)

    for i in range(len(self.res_list)):
      x = self.res_list[i](x)
    for i in range(len(self.deconv_list)):
      x = self.deconv_list[i](x)
    
    x = self.output_layer(x)
    return x

class Stage2_discriminator(tf.keras.Model):
  def __init__(self):
    super(Stage2_discriminator, self).__init__()
    self.input_layer = discriminator_Input(filters=64, strides=1)
    self.middle_layer_list1 = [
      discriminator_Middle(kernel_size=5, filters=128, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=256, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=512, strides=2, padding='valid'),
    ]
    self.middle_layer_list2 = [
      discriminator_Middle(kernel_size=1, filters=512, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=1024, strides=2, padding='valid'),
    ]
    self.output_layer = discriminator_Output(with_activation=False)

  def call(self, text_embedding, x):
    # print('text shape: {}'.format(text.shape))
    # print('x shape: {}'.format(x.shape))
    code = text_embedding
    code = tf.expand_dims(code, axis=1)
    code = tf.expand_dims(code, axis=2)

    x = self.input_layer(x)
    for i in range(len(self.middle_layer_list1)):
      x = self.middle_layer_list1[i](x)
    ones = tf.ones(shape=[1, x.shape[1], x.shape[2], 1], dtype=x.dtype)
    image_text = ones * code
    x = tf.concat([x, image_text], axis=-1)
    for i in range(len(self.middle_layer_list2)):
      x = self.middle_layer_list2[i](x)
    x = self.output_layer(x)
    return x

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

def get_gan(num_tokens):
  Stage1_Dense_mu_sigma = generate_condition(256*2)
  Stage2_Dense_mu_sigma = generate_condition(256 * 2)
  Embedding = embedding(num_encoder_tokens=num_tokens, embedding_dim=256, latent_dim=128)
  Stage1_Generator = Stage1_generator()
  Stage1_Discriminator = Stage1_discriminator()
  Stage2_Generator = Stage2_generator()
  Stage2_Discriminator = Stage2_discriminator()
  Generator = [Stage1_Generator, Stage2_Generator]
  Discriminator = [Stage1_Discriminator, Stage2_Discriminator]
  gen_name = 'Stack_GAN_lstm'
  return Generator, Discriminator, Embedding, Stage1_Dense_mu_sigma, Stage2_Dense_mu_sigma, gen_name


