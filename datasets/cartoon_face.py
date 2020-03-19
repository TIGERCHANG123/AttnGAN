import tensorflow as tf
import numpy as np
import os
import cv2

class face_dataset():
    def __init__(self, root, batch_size):
        self.file_path = root + '/datasets/cartoon_face/faces'
        image_width = 64
        self.batch_size = batch_size
        self.rate = image_width / 96
        self.file_list = os.listdir(self.file_path)
        self.name = 'cartoon_face'
    def generator(self):
        for name in self.file_list:
            img = cv2.imread('{}/{}'.format(self.file_path, name), 1)
            height, width = img.shape[:2]
            size = (int(width*self.rate), int(height*self.rate))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            yield img
    def parse(self, x):
        x = tf.cast(x, tf.float32)
        x = x/255 * 2 - 1
        return x
    def get_train_dataset(self):
        train = tf.data.Dataset.from_generator(self.generator, output_types=tf.int64)
        train = train.map(self.parse).shuffle(1000).batch(self.batch_size)
        return train

class noise_generator():
    def __init__(self, noise_dim, digit_dim, batch_size):
        self.noise_dim = noise_dim
        self.digit_dim = digit_dim
        self.batch_size = batch_size
    def get_noise(self):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        noise = tf.cast(noise, tf.float32)
        auxi_dict = np.random.multinomial(1, self.digit_dim * [float(1.0 / self.digit_dim)],size=[self.batch_size])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        return noise, auxi_dict

    def get_fixed_noise(self, num):
        noise = tf.random.normal([1, self.noise_dim])
        noise = tf.cast(noise, tf.float32)

        auxi_dict = np.array([num])
        auxi_dict = tf.convert_to_tensor(auxi_dict)
        auxi_dict = tf.one_hot(auxi_dict, depth=self.digit_dim)
        auxi_dict = tf.cast(auxi_dict, tf.float32)
        return noise, auxi_dict