# -*- coding:utf-8 -*-
import os
import getopt
import sys
import tensorflow as tf
from Stack_GAN import get_gan
from show_pic import draw
import fid
from Train import train_one_epoch
from datasets.oxford_102_flowers import oxford_102_flowers_dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# root = '/home/tigerc'
root = '/content/drive/My Drive'
dataset_root = '/content'
# dataset_root = '/home/tigerc'
temp_root = root+'/temp'

def main(continue_train, train_time, train_epoch):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    noise_dim = 100
    batch_size = 64

    dataset = oxford_102_flowers_dataset(dataset_root,batch_size = batch_size)
    [Stage1_generator, Stage2_generator], [Stage1_discriminator, Stage2_discriminator], embedding_model, model_name = get_gan(dataset.num_tokens)

    model_dataset = model_name + '-' + dataset.name

    train_dataset = dataset.get_train_dataset()
    pic = draw(10, temp_root, model_dataset, train_time=train_time)
    Stage1_generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    Stage1_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    Stage2_generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    Stage2_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    checkpoint_path = temp_root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(Stage1_genetator_optimizer=Stage1_generator_optimizer,
    Stage1_discriminator_optimizer=Stage1_discriminator_optimizer,
    Stage1_generator=Stage1_generator, Stage1_discriminator=Stage1_discriminator, 
    Stage2_genetator_optimizer=Stage2_generator_optimizer, Stage2_discriminator_optimizer=Stage2_discriminator_optimizer,
    Stage2_generator=Stage2_generator, Stage2_discriminator=Stage2_discriminator, embedding=embedding_model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
    if ckpt_manager.latest_checkpoint and continue_train:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')

    train = train_one_epoch(model=[Stage1_generator, Stage1_discriminator, Stage2_generator, Stage2_discriminator, embedding_model], 
              optimizers=[Stage1_generator_optimizer, Stage1_discriminator_optimizer, Stage2_generator_optimizer, Stage2_discriminator_optimizer],
              train_dataset=train_dataset, metrics=[gen_loss, disc_loss], noise_dim=noise_dim, gp=20)

    # pic.save_created_pic(generator_model, 8, noise_dim, 0, dataset.get_random_text, dataset.text_decoder)
    # pic.show_created_pic(generator_model, 8, noise_dim, dataset.get_random_text, dataset.text_decoder)
    for epoch in range(train_epoch):
        train.train(epoch=epoch, pic=pic, text_generator=dataset.get_random_text)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic([Stage1_generator, Stage2_generator, embedding_model], 
        8, noise_dim, epoch, dataset.get_random_text, dataset.text_decoder)
    pic.show_created_pic([Stage1_generator, Stage2_generator, embedding_model], 
    8, noise_dim, dataset.get_random_text, dataset.text_decoder)

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
