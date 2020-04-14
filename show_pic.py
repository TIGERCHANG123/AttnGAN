import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from time import sleep
import os
import tensorflow as tf
import cv2

class draw:
  batch_list = []
  loss_list = []
  acc_list = []
  i = 0
  def __init__(self, pic_size, root, model_dataset, train_time):
    rcParams['figure.figsize']=pic_size, pic_size
    self.pic_path = root + '/temp_pic/' + model_dataset
    self.pic_save_path = root + '/temp_pic_save/' + model_dataset
    self.generated_small_pic_path = root + '/generated_pic/' + model_dataset + '/small'
    self.generated_large_pic_path = root + '/generated_pic/' + model_dataset + '/large'
    self.train_time = train_time

    if not (os.path.exists(self.pic_path)):
        os.makedirs(self.pic_path)
    if not (os.path.exists(self.pic_save_path)):
        os.makedirs(self.pic_save_path)
    if not (os.path.exists(self.generated_small_pic_path)):
        os.makedirs(self.generated_small_pic_path)
    if not (os.path.exists(self.generated_large_pic_path)):
      os.makedirs(self.generated_large_pic_path)

    self.fig = plt.figure(figsize=(12, 4))
    self.batch_list = []
    self.train_loss_list = []
    self.train_acc_list = []
    self.i = 0
    self.large_pic_list=[]
    self.small_pic_list=[]
  def add(self, train_log):
    if len(self.batch_list) != 0:
      self.i = self.batch_list[-1] + 1
    else:
      self.i = self.i+1
    self.batch_list.append(self.i)
    self.train_loss_list.append(train_log[0])
    self.train_acc_list.append(train_log[1])
  def add_history(self, history):
      if len(self.batch_list) != 0:
          self.i = self.batch_list[-1] + len(history)
      else:
          self.i = self.i + len(history)
      self.batch_list.append(self.i)
      self.train_loss_list.append(history['loss'])
      self.train_acc_list.append(history['accuracy'])
  def save(self):
    NetPath = self.pic_save_path
    np.save(NetPath+'/b.npy', np.array(self.batch_list))
    np.save(NetPath+'/train_loss.npy', np.array(self.train_loss_list))
    np.save(NetPath+'/train_acc.npy', np.array(self.train_acc_list))
  def load(self, NetPath):
    self.batch_list = np.load(NetPath+'/b.npy').tolist()
    self.train_loss_list = np.load(NetPath+'/train_loss.npy').tolist()
    self.train_acc_list = np.load(NetPath+'/train_acc.npy').tolist()
  def close(self, time):
      sleep(time)
      plt.close()
  def show(self):
    file_path = self.pic_path
    plt.clf()
    ax1 = self.fig.add_subplot(121)
    ax2 = self.fig.add_subplot(122)
    ax1.plot(self.batch_list, self.train_loss_list, label = 'train loss', color = 'red')
    ax2.plot(self.batch_list, self.train_acc_list, label = 'train acc', color = 'red')
    bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
    ax1.annotate("%s" % self.train_loss_list[-1], xy=(self.i, self.train_loss_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    plt.annotate("%s" % self.train_acc_list[-1], xy=(self.i, self.train_acc_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    ax1.set(xlabel='batches',ylabel='loss', title = 'gen_loss')
    ax2.set(xlabel='batches',ylabel='loss', title = 'disc_loss')

    plt.savefig(file_path+'/{}_{}.png'.format(self.train_time, str(self.i)))
    # thread1 = Thread(target=self.close, args=(1,))
    # thread1.start()
    # plt.show()
  def show_image(self, image):
    plt.imshow(image)
    plt.show()

  def save_created_pic(self, models, pic_num, noise_dim, epoch, mid_epoch, text_generator, text_decoder):
    Generator, Discriminator, embedding, Dense_mu_sigma = models
    text = text_generator(pic_num)
    sentence = []
    for i in range(text.shape[0]):
      s = text_decoder(text[i])
      s = s.split(' ')
      s = '_'.join(s)
      sentence.append(s)
    x = tf.convert_to_tensor(np.random.rand(pic_num, noise_dim), dtype=tf.float32)
    # print('x type: {}'.format(x.dtype))
    # print('text type: {}'.format(text.dtype))
    sequence, embedding_code = embedding(text)
    mu, sigma = Dense_mu_sigma(embedding_code)
    epsilon = tf.compat.v1.random.truncated_normal(tf.shape(mu))
    stddev = tf.exp(sigma)
    text1 = mu + stddev * epsilon
    # text1 = embedding_code
    y0, y1 = Generator(sequence, text1, x)
    y0=tf.squeeze(y0)
    y1=tf.squeeze(y1)
    y0 = (y0+1)/2
    y1 = (y1+1)/2
    for i in range(pic_num):
      # b, g, r = cv2.split((y0[i].numpy()*255).astype(np.uint8))
      # img = cv2.merge([r, g, b])
      img = (y0[i].numpy() * 255).astype(np.uint8)
      cv2.imwrite(self.generated_small_pic_path+'/{}_{}_{}_{}.png'.format(self.train_time, epoch, i, sentence[i]),img)
      # b, g, r = cv2.split((y1[i].numpy() * 255).astype(np.uint8))
      # img = cv2.merge([r, g, b])
      img = (y1[i].numpy() * 255).astype(np.uint8)
      cv2.imwrite(self.generated_large_pic_path + '/{}_{}_{}_{}.png'.format(self.train_time, epoch, i, sentence[i]),img)
      # self.large_pic_list.append(self.generated_small_pic_path+'/{}_{}_{}_{}.png'.format(self.train_time, epoch, i, sentence[i]))
      # self.small_pic_list.append(self.generated_large_pic_path + '/{}_{}_{}_{}.png'.format(self.train_time, epoch, i, sentence[i]))
    # if len(self.large_pic_list) > 50:
    #   for i in range(pic_num):
    #     pic_path = self.large_pic_list.pop(0)
    #     os.remove(pic_path)
    #     pic_path = self.small_pic_list.pop(0)
    #     os.remove(pic_path)
    return