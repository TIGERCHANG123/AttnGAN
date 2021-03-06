import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim, gp):
        self.Generator, self.Discriminator,  self.embedding, self.Dense_mu_sigma = model
        self.Stage1_generator_optimizer, self.Stage1_discriminator_optimizer, self.Stage2_generator_optimizer, self.Stage2_discriminator_optimizer = optimizers

        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.gp = gp

        self.fake_loss = 0
        self.real_loss = 0
        self.grad_penalty = 0

    def discriminator_loss(self, real_output, fake_output1, fake_output2):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss1 = cross_entropy(tf.zeros_like(fake_output1), fake_output1)
        fake_loss2 = cross_entropy(tf.zeros_like(fake_output2), fake_output2)
        total_loss = real_loss + (fake_loss1 + fake_loss2)/2
        return total_loss, real_loss, fake_loss1, fake_loss2

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def KL_loss(self, mu, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

    def train_step(self, noise, images_1, images_2, text, text_generator):
        with tf.GradientTape() as Stage1_gen_tape, tf.GradientTape() as Stage1_disc_tape, tf.GradientTape() as Stage2_disc_tape:
            sequence, memory_state = self.embedding(text)
            mu_1, sigma_1 = self.Dense_mu_sigma(memory_state)
            KL_loss = self.KL_loss(mu_1, sigma_1)
            epsilon = tf.compat.v1.random.truncated_normal(tf.shape(mu_1))
            stddev = tf.exp(sigma_1)
            text = mu_1 + stddev * epsilon
            # text = embedding_code
            small_gen, large_gen = self.Generator(sequence, text, noise)
            small_real, large_real = self.Discriminator(memory_state, images_1, images_2)
            small_fake1, large_fake1 = self.Discriminator(memory_state, small_gen, large_gen)

            fake_text = text_generator(images_1.shape[0])
            fake_sequence, fake_text = self.embedding(fake_text)
            small_fake2, large_fake2 = self.Discriminator(fake_text, images_1, images_2)
            # loss calculation
            Stage1_disc_loss, Stage1_real_loss, Stage1_fake_loss1, Stage1_fake_loss2 \
                = self.discriminator_loss(small_real, small_fake1, small_fake2)

            Stage2_disc_loss, Stage2_real_loss, Stage2_fake_loss1, Stage2_fake_loss2 \
                = self.discriminator_loss(large_real, large_fake1, large_fake2)

            Stage1_gen_loss = self.generator_loss(small_fake1) + KL_loss
            Stage2_gen_loss = self.generator_loss(large_fake1)
            gen_loss_total = Stage1_gen_loss + Stage2_gen_loss

            self.disc_loss(Stage2_disc_loss + Stage1_disc_loss)
            self.gen_loss(gen_loss_total)

            Stage1_disc_loss = self.Stage1_discriminator_optimizer.get_scaled_loss(Stage1_disc_loss)
            Stage2_disc_loss = self.Stage2_discriminator_optimizer.get_scaled_loss(Stage2_disc_loss)
            gen_loss_total = self.Stage1_generator_optimizer.get_scaled_loss(gen_loss_total)
        Stage1_dist_variables = [v for v in self.Discriminator.trainable_variables if 'h0' in v.name]
        gradients_of_discriminator = Stage1_disc_tape.gradient(Stage1_disc_loss,Stage1_dist_variables)
        gradients_of_discriminator = self.Stage1_discriminator_optimizer.get_unscaled_gradients(gradients_of_discriminator)
        self.Stage1_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,Stage1_dist_variables))

        Stage2_dist_variables = [v for v in self.Discriminator.trainable_variables if 'h1' in v.name]
        gradients_of_discriminator = Stage2_disc_tape.gradient(Stage2_disc_loss, Stage2_dist_variables)
        gradients_of_discriminator = self.Stage2_discriminator_optimizer.get_unscaled_gradients(gradients_of_discriminator)
        self.Stage2_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, Stage2_dist_variables))

        gen_variables = self.Generator.trainable_variables + self.Dense_mu_sigma.trainable_variables
        gradients_of_generator = Stage1_gen_tape.gradient(gen_loss_total, gen_variables)
        gradients_of_generator= self.Stage1_generator_optimizer.get_unscaled_gradients(gradients_of_generator)
        self.Stage1_generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_variables))

    def train_step_0(self, noise, images_1, images_2, text, text_generator):
        with tf.GradientTape() as Stage1_gen_tape, tf.GradientTape() as Stage1_disc_tape, tf.GradientTape() as Stage2_disc_tape:
            sequence, memory_state = self.embedding(text)
            # mu_1, sigma_1 = self.Dense_mu_sigma(memory_state)
            # KL_loss = self.KL_loss(mu_1, sigma_1)
            # epsilon = tf.compat.v1.random.truncated_normal(tf.shape(mu_1))
            # stddev = tf.exp(sigma_1)
            text = memory_state
            # text = embedding_code
            small_gen = self.Generator(sequence, text, noise)
            small_real = self.Discriminator(memory_state, images_1)
            small_fake1 = self.Discriminator(memory_state, small_gen)

            fake_text = text_generator(images_1.shape[0])
            fake_sequence, fake_text = self.embedding(fake_text)
            small_fake2 = self.Discriminator(fake_text, images_1)
            # loss calculation
            Stage1_disc_loss, Stage1_real_loss, Stage1_fake_loss1, Stage1_fake_loss2 \
                = self.discriminator_loss(small_real, small_fake1, small_fake2)

            Stage1_gen_loss = self.generator_loss(small_fake1)
            gen_loss_total = Stage1_gen_loss

            self.disc_loss(Stage1_disc_loss)
            self.gen_loss(gen_loss_total)

            Stage1_disc_loss = self.Stage1_discriminator_optimizer.get_scaled_loss(Stage1_disc_loss)
            gen_loss_total = self.Stage1_generator_optimizer.get_scaled_loss(gen_loss_total)
        Stage1_dist_variables = [v for v in self.Discriminator.trainable_variables if 'h0' in v.name]
        gradients_of_discriminator = Stage1_disc_tape.gradient(Stage1_disc_loss,Stage1_dist_variables)
        gradients_of_discriminator = self.Stage1_discriminator_optimizer.get_unscaled_gradients(gradients_of_discriminator)
        self.Stage1_discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,Stage1_dist_variables))

        gen_variables = [v for v in self.Generator.trainable_variables if 'h0' in v.name] + self.Dense_mu_sigma.trainable_variables
        gradients_of_generator = Stage1_gen_tape.gradient(gen_loss_total, gen_variables)
        gradients_of_generator= self.Stage1_generator_optimizer.get_unscaled_gradients(gradients_of_generator)
        self.Stage1_generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_variables))

    def train(self, epoch, mid_epoch, pic, text_generator):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (image_1, image_2, text)) in enumerate(self.train_dataset):
            noise = tf.random.normal([image_1.shape[0], self.noise_dim], dtype=tf.float32)
            self.train_step(noise, image_1, image_2, text, text_generator)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result()))