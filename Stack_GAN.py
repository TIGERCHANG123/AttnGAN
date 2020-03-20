from Stack_GAN_Block import *

class embedding(tf.keras.Model):
  def __init__(self, num_encoder_tokens, embedding_dim):
    super(embedding, self).__init__()
    self.embedding = layers.Embedding(num_encoder_tokens, embedding_dim)
  def call(self, text):
    code = self.embedding(text)
    return code

class Stage1_generator(tf.keras.Model):
  def __init__(self):
    super(Stage1_generator, self).__init__()
    self.encoder = Encoder(latent_dim=128)
    self.input_layer = generator_Input(shape=[4, 4, 1024])

    self.middle_layer_list = [
      # generator_Middle(filters=1024, strides=2, padding='same'),  # 1024*4*4
      generator_Middle(filters=512, strides=2, padding='same'),#1024*4*4
      generator_Middle(filters=256, strides=2, padding='same'),#512*8*8
      generator_Middle(filters=128, strides=2, padding='same'),#256*16*16
    ]

    self.output_layer = generator_Output(image_depth=3, strides=2, padding='same')#3*32*32
  def call(self, text_embedding, noise):
    x = self.encoder(text_embedding)
    x = tf.concat([noise, x], axis=-1)
    x = self.input_layer(x)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
    x = self.output_layer(x)
    return x

class Stage1_discriminator(tf.keras.Model):
  def __init__(self):
    super(Stage1_discriminator, self).__init__()
    self.encoder = Encoder(latent_dim=128)
    self.input_layer = discriminator_Input(filters=128, strides=1)
    self.middle_layer_list1 = [
      discriminator_Middle(kernel_size=5, filters=256, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=512, strides=2, padding='valid'),
    ]
    self.middle_layer_list2 = [
      discriminator_Middle(kernel_size=1, filters=1024, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=1024, strides=2, padding='valid'),
    ]
    self.output_layer = discriminator_Output(with_activation=False)

  def call(self, text_embedding, x):
    # print('text shape: {}'.format(text.shape))
    # print('x shape: {}'.format(x.shape))
    code = self.encoder(text_embedding)
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
    print('Stage1 discriminator x shape: {}'.format(x.shape))
    x = self.output_layer(x)
    return x

class Stage2_generator(tf.keras.Model):
  def __init__(self):
    super(Stage2_generator, self).__init__()
    self.encoder = Encoder(latent_dim=128)
    
    self.conv_list = [
      discriminator_Middle(kernel_size=5, filters=256, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=512, strides=2, padding='valid'),
    ]
    self.deconv_list = [
      # generator_Middle(filters=1024, strides=2, padding='same'),  # 1024*4*4
      generator_Middle(filters=512, strides=2, padding='same'),#1024*4*4
      generator_Middle(filters=256, strides=2, padding='same'),#512*8*8
      generator_Middle(filters=128, strides=2, padding='same'),#256*16*16
    ]

    self.output_layer = generator_Output(image_depth=3, strides=2, padding='same')#3*32*32
  def call(self, text_embedding, x):
    # print('text shape: {}'.format(text.shape))
    # print('x shape: {}'.format(x.shape))
    for i in range(len(self.conv_list)):
          x = self.conv_list[i](x)

    code = self.encoder(text_embedding)
    code = tf.expand_dims(code, axis=1)
    code = tf.expand_dims(code, axis=2)
    ones = tf.ones(shape=[1, x.shape[1], x.shape[2], 1], dtype=x.dtype)
    image_text = ones * code
    x = tf.concat([x, image_text], axis=-1)

    for i in range(len(self.deconv_list)):
      x = self.deconv_list[i](x)
    
    x = self.output_layer(x)
    return x

class Stage2_discriminator(tf.keras.Model):
  def __init__(self):
    super(Stage2_discriminator, self).__init__()
    self.encoder = Encoder(latent_dim=128)
    self.input_layer = discriminator_Input(filters=128, strides=1)
    self.middle_layer_list1 = [
      discriminator_Middle(kernel_size=5, filters=256, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=512, strides=2, padding='valid'),
    ]
    self.middle_layer_list2 = [
      discriminator_Middle(kernel_size=1, filters=1024, strides=2, padding='valid'),
      discriminator_Middle(kernel_size=5, filters=1024, strides=2, padding='valid'),
    ]
    self.output_layer = discriminator_Output(with_activation=False)

  def call(self, text_embedding, x):
    # print('text shape: {}'.format(text.shape))
    # print('x shape: {}'.format(x.shape))
    code = self.encoder(text_embedding)
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

def get_gan(num_tokens):
  Embedding = embedding(num_encoder_tokens=num_tokens, embedding_dim=256)
  Stage1_Generator = Stage1_generator()
  Stage1_Discriminator = Stage1_discriminator()
  Stage2_Generator = Stage2_generator()
  Stage2_Discriminator = Stage2_discriminator()
  Generator = [Stage1_Generator, Stage2_Generator]
  Discriminator = [Stage1_Discriminator, Stage2_Discriminator]
  gen_name = 'Stack_GAN'
  return Generator, Discriminator, Embedding, gen_name


