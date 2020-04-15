# -*- coding:utf-8 -*-
import os
import getopt
import sys
import tensorflow as tf
from AttnGAN import get_gan
from show_pic import draw
import fid
from Train import train_one_epoch
from datasets.oxford_102_flowers import oxford_102_flowers_dataset
from datasets.CUB import CUB_dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

root = '/home/tigerc'
# root = '/content/drive/My Drive'
# dataset_root = '/content'
dataset_root = '/home/tigerc'
temp_root = root+'/temp'

def main(continue_train, train_time, train_epoch, mid_epoch):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    noise_dim = 100
    batch_size = 48

    # dataset = oxford_102_flowers_dataset(dataset_root,batch_size = batch_size)
    dataset = CUB_dataset(dataset_root,batch_size = batch_size)
    Dense_mu_sigma_model, embedding_model,  Generator, Discriminator, model_name = get_gan(dataset.num_tokens)

    model_dataset = model_name + '-' + dataset.name

    train_dataset = dataset.get_train_dataset()
    pic = draw(10, temp_root, model_dataset, train_time=train_time)
    lr = 2e-4
    Stage1_generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    Stage1_discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    Stage2_generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    Stage2_discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

    checkpoint_path = temp_root + '/temp_model_save/' + model_dataset
    embedding_checkpoint_path = './embedding_model/' + dataset.name
    ckpt = tf.train.Checkpoint(Stage1_genetator_optimizer=Stage1_generator_optimizer,
    Stage1_discriminator_optimizer=Stage1_discriminator_optimizer,
    Stage2_genetator_optimizer=Stage2_generator_optimizer, Stage2_discriminator_optimizer=Stage2_discriminator_optimizer,
    Generator=Generator, Discriminator=Discriminator,Dense_mu_sigma_model=Dense_mu_sigma_model, embedding_model=embedding_model)
    embedding_ckpt = tf.train.Checkpoint(embedding=embedding_model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
    embedding_ckpt_manager = tf.train.CheckpointManager(embedding_ckpt, embedding_checkpoint_path, max_to_keep=1)
    
    if ckpt_manager.latest_checkpoint and continue_train:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    # embedding_ckpt.restore(dataset_root+'/AttnGAN/embedding_model/'+dataset.name+'/ckpt-223')
    embedding_ckpt.restore('./embedding_model/' + dataset.name + '/ckpt-173')
    # embedding_ckpt.restore('./embedding_model/' + dataset.name + '/ckpt-223')
    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')

    train = train_one_epoch(model=[Generator, Discriminator, embedding_model, Dense_mu_sigma_model],
              optimizers=[Stage1_generator_optimizer, Stage1_discriminator_optimizer, Stage2_generator_optimizer, Stage2_discriminator_optimizer],
              train_dataset=train_dataset, metrics=[gen_loss, disc_loss], noise_dim=noise_dim, gp=20)

    # pic.save_created_pic([Stage1_generator, Stage2_generator, embedding_model],
    #                      8, noise_dim, 0, dataset.get_random_text, dataset.text_decoder)
    for epoch in range(train_epoch):
        train.train(epoch=epoch, mid_epoch=mid_epoch, pic=pic, text_generator=dataset.get_random_text)
        pic.show()
        # if (epoch + 1) % 100 == 0:
        #     lr *= 0.5
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        try:
            pic.save_created_pic([Generator, Discriminator, embedding_model, Dense_mu_sigma_model],
                8, noise_dim, epoch, mid_epoch, dataset.get_random_text, dataset.text_decoder)
        except:
            continue

    # # fid score
    # gen = generator_model
    # noise = noise_generator(noise_dim, 10, batch_size, dataset.total_pic_num//batch_size)()
    # real_images = dataset.get_train_dataset()
    # fd = fid.FrechetInceptionDistance(gen, (-1, 1), [128, 128, 3])
    # gan_fid, gan_is = fd(iter(real_images), noise, batch_size=batch_size, num_batches_real=dataset.total_pic_num//batch_size)
    # print('fid score: {}, inception score: {}'.format(gan_fid, gan_is))

    return
if __name__ == '__main__':
    continue_train = False
    train_time = 0
    epoch = 500
    mid_epoch = 300
    try:
        opts, args = getopt.getopt(sys.argv[1:], '-c-t:-e:-m:', ['continue', 'time=', 'epoch=', 'mid_epoch='])
        for op, value in opts:
            print(op, value)
            if op in ('-c', '--continue'):
                continue_train = True
            elif op in ('-t', '--time'):
                train_time = int(value)
            elif op in ('-e', '--epoch'):
                epoch = int(value)
            elif op in ('-m', '--mid_epoch'):
                mid_epoch = int(value)
    except:
        print('wrong input!')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main(continue_train=continue_train, train_time=train_time, train_epoch=epoch, mid_epoch=mid_epoch)
