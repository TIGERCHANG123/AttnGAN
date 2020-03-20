import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim, gp):
        self.Stage1_generator, self.Stage1_discriminator, self.Stage2_generator, self.Stage2_discriminator, self.embedding = model
        self.Stage1_generator_optimizer, self.Stage1_discriminator_optimizer, self.Stage2_generator_optimizer, self.Stage2_discriminator_optimizer = optimizers

        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.gp = gp

        self.fake_loss = 0
        self.real_loss = 0
        self.grad_penalty = 0

    def discriminator_loss(self, real_output, fake_output1, fake_output2):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss1 = cross_entropy(tf.zeros_like(fake_output1), fake_output1)
        fake_loss2 = cross_entropy(tf.zeros_like(fake_output2), fake_output2)
        total_loss = real_loss + fake_loss1 + fake_loss2
        return total_loss, real_loss, fake_loss1, fake_loss2

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_step(self, noise, images_1, images_2, text_1, text_2, text_generator):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            text0 = self.embedding(text_1)
            text1 = self.embedding(text_2)
            text = (text0 + text1)/2
            generated_images = self.Stage1_generator(text, noise, training=True)
            real_output = self.Stage1_discriminator(text, images_1, training=True)
            fake_output1 = self.Stage1_discriminator(text, generated_images, training=True)
            fake_text = text_generator(images_1.shape[0])
            fake_text = self.embedding(fake_text)
            fake_output2 = self.Stage1_discriminator(fake_text, images_1, training=True)

            disc_loss, real_loss, fake_loss1, fake_loss2 = self.discriminator_loss(real_output, fake_output1, fake_output2)
            gen_loss = self.generator_loss(fake_output1)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.Stage1_generator.trainable_variables+self.embedding.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Stage1_discriminator.trainable_variables+self.embedding.trainable_variables)
        self.Stage1_generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Stage1_generator.trainable_variables+self.embedding.trainable_variables))
        self.Stage1_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.Stage1_discriminator.trainable_variables+self.embedding.trainable_variables))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.Stage2_generator(text, generated_images, training=True)
            real_output = self.Stage2_discriminator(text, images_2, training=True)
            fake_output1 = self.Stage2_discriminator(text, generated_images, training=True)
            fake_text = text_generator(images_1.shape[0])
            fake_text = self.embedding(fake_text)
            fake_output2 = self.Stage2_discriminator(fake_text, images_2, training=True)

            disc_loss, real_loss, fake_loss1, fake_loss2 = self.discriminator_loss(real_output, fake_output1, fake_output2)
            gen_loss = self.generator_loss(fake_output1)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.Stage2_generator.trainable_variables+self.embedding.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Stage2_discriminator.trainable_variables+self.embedding.trainable_variables)
        self.gen_loss(gen_loss)
        self.disc_loss(disc_loss)
        self.Stage2_generator_optimizer.apply_gradients(zip(gradients_of_generator, self.Stage2_generator.trainable_variables+self.embedding.trainable_variables))
        self.Stage2_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.Stage2_discriminator.trainable_variables+self.embedding.trainable_variables))
    def train(self, epoch,  pic, text_generator):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (image_1, image_2, text1, text2)) in enumerate(self.train_dataset):
            noise = tf.random.normal([image_1.shape[0], self.noise_dim], dtype=tf.float32)
            self.train_step(noise, image_1, image_2, text1, text2, text_generator)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}, grad penalty: {}, real loss: {}, fake loss: {}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result(), self.grad_penalty, self.real_loss, self.fake_loss))