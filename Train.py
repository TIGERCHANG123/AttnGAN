import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim, gp):
        self.Stage1_generator, self.Stage1_discriminator, self.Stage2_generator, self.Stage2_discriminator\
            , self.embedding, self.Stage1_Dense_mu_sigma, self.Stage2_Dense_mu_sigma = model
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
        total_loss = real_loss + (fake_loss1 + fake_loss2)/2
        return total_loss, real_loss, fake_loss1, fake_loss2

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def KL_loss(self, mu, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

    def train_step(self, noise, images_1, images_2, text, text_generator, Stage1_training):
        with tf.GradientTape() as Stage1_gen_tape, tf.GradientTape() as Stage1_disc_tape, \
                tf.GradientTape() as Stage2_gen_tape, tf.GradientTape() as Stage2_disc_tape:
            embedding_code = self.embedding(text, training=Stage1_training)
            mu_1, sigma_1 = self.Stage1_Dense_mu_sigma(embedding_code, training=Stage1_training)
            Stage1_KL_loss = self.KL_loss(mu_1, sigma_1)
            epsilon = tf.compat.v1.random.truncated_normal(tf.shape(mu_1))
            stddev = tf.exp(sigma_1)
            text = mu_1 + stddev * epsilon
            generated_images = self.Stage1_generator(text, noise, training=Stage1_training)
            real_output = self.Stage1_discriminator(embedding_code, images_1, training=Stage1_training)
            fake_output1 = self.Stage1_discriminator(embedding_code, generated_images, training=Stage1_training)
            fake_text = text_generator(images_1.shape[0])
            fake_text = self.embedding(fake_text, training=Stage1_training)
            fake_output2 = self.Stage1_discriminator(fake_text, images_1, training=Stage1_training)
            if Stage1_training:
                Stage1_disc_loss, Stage1_real_loss, Stage1_fake_loss1, Stage1_fake_loss2 \
                    = self.discriminator_loss(real_output, fake_output1, fake_output2)
                Stage1_gen_loss = self.generator_loss(fake_output1) + Stage1_KL_loss
            else:
                mu_2, sigma_2 = self.Stage2_Dense_mu_sigma(embedding_code, training=not Stage1_training)
                Stage2_KL_loss = self.KL_loss(mu_2, sigma_2)
                epsilon = tf.compat.v1.random.truncated_normal(tf.shape(mu_2))
                stddev = tf.exp(sigma_2)
                text = mu_2 + stddev * epsilon
                generated_images = self.Stage2_generator(text, generated_images, training=not Stage1_training)
                real_output = self.Stage2_discriminator(embedding_code, images_2, training=not Stage1_training)
                fake_output1 = self.Stage2_discriminator(embedding_code, generated_images, training=not Stage1_training)
                fake_text = text_generator(images_1.shape[0])
                fake_text = self.embedding(fake_text)
                fake_output2 = self.Stage2_discriminator(fake_text, images_2, training=not Stage1_training)

                Stage2_disc_loss, Stage2_real_loss, Stage2_fake_loss1, Stage2_fake_loss2 \
                    = self.discriminator_loss(real_output, fake_output1, fake_output2)
                Stage2_gen_loss = self.generator_loss(fake_output1) + Stage2_KL_loss
        if Stage1_training:
            gradients_of_generator = Stage1_gen_tape.gradient(Stage1_gen_loss,
                                                              self.Stage1_generator.trainable_variables+
                                                              self.embedding.trainable_variables+
                                                              self.Stage1_Dense_mu_sigma.trainable_variables
                                                              )
            self.Stage1_generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                                self.Stage1_generator.trainable_variables+
                                                                self.embedding.trainable_variables+
                                                                self.Stage1_Dense_mu_sigma.trainable_variables))
            gradients_of_discriminator = Stage1_disc_tape.gradient(Stage1_disc_loss,
                                                                   self.Stage1_discriminator.trainable_variables +
                                                                   self.embedding.trainable_variables)
            self.Stage1_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                                    self.Stage1_discriminator.trainable_variables+
                                                                    self.embedding.trainable_variables))
            self.gen_loss(Stage1_gen_loss)
            self.disc_loss(Stage1_disc_loss)
        else:
            gradients_of_generator = Stage2_gen_tape.gradient(Stage2_gen_loss,
                                                              self.Stage2_generator.trainable_variables+
                                                              # self.embedding.trainable_variables +
                                                              self.Stage2_Dense_mu_sigma.trainable_variables
                                                              )
            self.Stage2_generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                                self.Stage2_generator.trainable_variables+
                                                                # self.embedding.trainable_variables +
                                                                self.Stage2_Dense_mu_sigma.trainable_variables))
            gradients_of_discriminator = Stage2_disc_tape.gradient(Stage2_disc_loss,
                                                                   # self.embedding.trainable_variables +
                                                                   self.Stage2_discriminator.trainable_variables)
            self.Stage2_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                                    # self.embedding.trainable_variables+
                                                                    self.Stage2_discriminator.trainable_variables))

            self.gen_loss(Stage2_gen_loss)
            self.disc_loss(Stage2_disc_loss)
    def train(self, epoch, mid_epoch, pic, text_generator):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (image_1, image_2, text)) in enumerate(self.train_dataset):
            noise = tf.random.normal([image_1.shape[0], self.noise_dim], dtype=tf.float32)
            self.train_step(noise, image_1, image_2, text, text_generator, epoch < mid_epoch)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result()))