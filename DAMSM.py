# -*- coding:utf-8 -*-
import os
import getopt
import sys
from show_pic import draw
from datasets.CUB import CUB_dataset
from AttnGAN import *
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np

root = '/content/drive/My Drive'
dataset_root = '/content'

# root = '/home/tigerc'
# dataset_root = '/home/tigerc'
temp_root = root+'/temp'

class attention(tf.keras.Model):
  def __init__(self, num_tokens, name):
    super(attention, self).__init__()
    self.dense = layers.Dense(units=num_tokens, name=name+'_dense')
  def call(self, e, f, gamma):
    print('e shape', e.shape)
    v = self.dense(tf.transpose(f, [0, 2, 1]))
    v = tf.transpose(v, [0, 2, 1])
    print('v shape', v.shape)
    s = tf.matmul(tf.transpose(e, [0, 2, 1]), v)
    print('s shape', s.shape)
    alpha = tf.nn.softmax(gamma*s, axis=2)
    c = tf.matmul(v, tf.transpose(alpha, [0, 2, 1]))
    print('c shape', c.shape)
    lc = tf.math.sqrt(tf.reduce_sum(c*c, axis=1))
    le = tf.math.sqrt(tf.reduce_sum(e*e, axis=1))
    cosine_similarity = tf.matmul(c, tf.transpose(e, [0, 2, 1]))/(lc*le)
    print('cosine similarity shape', cosine_similarity.shape)
    return cosine_similarity

class attention_(tf.keras.Model):
    def __init__(self, num_tokens, name):
        super(attention_, self).__init__()
        self.dense = layers.Dense(units=num_tokens, name=name + '_dense')
    def call(self, e_, f_):
        print('e shape', e_.shape)
        v_ = self.dense(f_)
        print('v shape', v_.shape)
        s_ = tf.matmul(e_, v_)
        print('s shape', s_.shape)
        lc_ = tf.math.sqrt(v_ * v_, axis=1)
        le_ = tf.math.sqrt(e_ * e_, axis=1)
        cosine_similarity = (v_ * e_) / (lc_ * le_)
        print('cosine similarity shape', cosine_similarity.shape)
        return cosine_similarity

def damsm_model(num_tokens):
  Attention1 = attention(num_tokens, 'attention')
  Attention2 = attention_(num_tokens, 'attention_')
  embedding_model = embedding(num_encoder_tokens=num_tokens, embedding_dim=256, latent_dim=128)
  gen_name = 'damsm'
  return Attention1, Attention2, embedding_model, gen_name

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics):
        self.Attention1, self.Attention2, self.embedding_model  = model
        self.optimizer = optimizers

        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer1 = image_model.layers[-1].output
        hidden_layer2 = image_model.layers[-1].output
        self.InceptionV3 = tf.keras.Model(new_input, [hidden_layer1, hidden_layer2])
        self.loss = metrics
        self.train_dataset = train_dataset
        self.gamma1 = 5
        self.gamma2 = 5
        self.gamma3 = 10

    def train_step(self, images_2, text):
        with tf.GradientTape() as tape:
            img=[]
            for i in range(images_2.shape[0]):
                img.append(cv2.resize(images_2[i].numpy(), [299, 299, 3]))
            img = np.asarray(img)
            img = tf.convert_to_tensor(img)
            print('img shape', img.shape)
            f, f_ = self.InceptionV3(img)
            e, e_ = self.embedding_model(text)
            cosine_similarity = self.Attention1(e, f, self.gamma1)
            cosine_similarity_ = self.Attention2(e_, f_)
            RQD = tf.math.pow(tf.math.log(tf.reduce_sum(tf.math.exp(self.gamma2 * cosine_similarity), axis=2)), 1 / self.gamma2)
            RDQ = tf.math.pow(tf.math.log(tf.reduce_sum(tf.math.exp(self.gamma2 * cosine_similarity), axis=1)), 1 / self.gamma2)
            print('RQD shape', RQD.shape)
            PQD = tf.nn.softmax(self.gamma3*RQD, axis=1)
            PDQ = tf.nn.softmax(self.gamma3*RDQ, axis=1)
            L_1w = -tf.reduce_sum(tf.math.log(PQD))
            L_2w = -tf.reduce_sum(tf.math.log(PDQ))

            PQD_ = tf.nn.softmax(self.gamma3*cosine_similarity_, axis=1)
            PDQ_ = tf.nn.softmax(self.gamma3*cosine_similarity_, axis=1)
            L_1s = -tf.reduce_sum(tf.math.log(PQD_))
            L_2s = -tf.reduce_sum(tf.math.log(PDQ_))

            loss = L_1w + L_2w + L_1s + L_2s
        self.loss(loss)
        variables = self.Attention1.variables + self.Attention2.variables + self.embedding_model.variables
        gradients_of_generator = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients_of_generator, variables))
    def train(self, epoch, pic):
        self.loss.reset_states()

        for (batch, (image_1, image_2, text)) in enumerate(self.train_dataset):
            self.train_step(image_2, text)
            pic.add([self.loss.result().numpy(), 0])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, damsm loss: {}'.format(epoch, self.loss.result()))

def main(continue_train, train_time, train_epoch):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    batch_size = 50

    dataset = CUB_dataset(dataset_root,batch_size = batch_size)
    Attention1, Attention2, embedding_model, model_name = damsm_model(dataset.num_tokens)

    model_dataset = model_name + '-' + dataset.name
    train_dataset = dataset.get_train_dataset()
    pic = draw(10, temp_root, model_dataset, train_time=train_time)
    damsm_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    attn_path = temp_root + '/temp_model_save/attn/' + model_dataset
    embedding_path = temp_root + '/temp_model_save/embedding/' + model_dataset

    attn_ckpt = tf.train.Checkpoint(damsm_optimizer=damsm_optimizer,
    Attention1=Attention1, Attention2=Attention2)
    embedding_ckpt = tf.train.Checkpoint(embedding=embedding_model)
    attn_ckpt_manager = tf.train.CheckpointManager(attn_ckpt, attn_path, max_to_keep=5)
    embedding_ckpt_manager = tf.train.CheckpointManager(embedding_ckpt, embedding_path, max_to_keep=5)
    if attn_ckpt_manager.latest_checkpoint and continue_train:
        attn_ckpt.restore(attn_ckpt_manager.latest_checkpoint)
        print('Latest attn checkpoint restored!!')
    if embedding_ckpt_manager.latest_checkpoint and continue_train:
        embedding_ckpt.restore(embedding_ckpt_manager.latest_checkpoint)
        print('Latest embedding checkpoint restored!!')
    loss = tf.keras.metrics.Mean(name='damsm loss')

    train = train_one_epoch(model=[Attention1, Attention2, embedding_model],optimizers=damsm_optimizer,train_dataset=train_dataset, metrics=loss)

    for epoch in range(train_epoch):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            attn_ckpt_manager.save()
            embedding_ckpt_manager.save()
    return

if __name__ == '__main__':
    continue_train = False
    train_time = 0
    epoch = 500
    try:
        opts, args = getopt.getopt(sys.argv[1:], '-c-t:-e:', ['continue', 'time=', 'epoch='])
        for op, value in opts:
            print(op, value)
            if op in ('-c', '--continue'):
                continue_train = True
            elif op in ('-t', '--time'):
                train_time = int(value)
            elif op in ('-e', '--epoch'):
                epoch = int(value)
    except:
        print('wrong input!')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main(continue_train=continue_train, train_time=train_time, train_epoch=epoch)