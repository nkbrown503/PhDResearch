# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:11:43 2022

@author: nbrow
"""

import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.activations import mish
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf

def loss_func(encoder_mu, encoder_log_variance):
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1000
            reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2, 3])
            return reconstruction_loss_factor * reconstruction_loss
    
        def vae_kl_loss(encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
            return kl_loss
    
        def vae_kl_loss_metric(y_true, y_predict):
            kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
            return kl_loss
    
        def vae_loss(y_true, y_predict):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
            kl_loss = vae_kl_loss(y_true, y_predict)
    
            loss = reconstruction_loss + kl_loss
            return loss
    
        return vae_loss
def VAE(y_size,x_size,num_channels,latent_space_dim):
    def loss_func(encoder_mu, encoder_log_variance):
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1000
            reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2, 3])
            return reconstruction_loss_factor * reconstruction_loss
    
        def vae_kl_loss(encoder_mu, encoder_log_variance):
            kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
            return kl_loss
    
        def vae_kl_loss_metric(y_true, y_predict):
            kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
            return kl_loss
    
        def vae_loss(y_true, y_predict):
            reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
            kl_loss = vae_kl_loss(y_true, y_predict)
    
            loss = reconstruction_loss + kl_loss
            return loss
    
        return vae_loss


    x = Input(shape=(y_size, x_size, num_channels), name="encoder_input")
    
    encoder_conv_layer1 = Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=1, name="encoder_conv_1")(x)
    encoder_norm_layer1 = BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
    encoder_activ_layer1 =ReLU()(encoder_norm_layer1)
    
    encoder_conv_layer2 = Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=1, name="encoder_conv_2")(encoder_activ_layer1)
    encoder_norm_layer2 = BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
    encoder_activ_layer2 = ReLU()(encoder_norm_layer2)
    
    encoder_conv_layer3 = Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2, name="encoder_conv_3")(encoder_activ_layer2)
    encoder_norm_layer3 = BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
    encoder_activ_layer3 = ReLU()(encoder_norm_layer3)
    
    encoder_conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2, name="encoder_conv_4")(encoder_activ_layer3)
    encoder_norm_layer4 = BatchNormalization(name="encoder_norm_4")(encoder_conv_layer4)
    encoder_activ_layer4 =ReLU()(encoder_norm_layer4)
    
    encoder_conv_layer5 = Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=1, name="encoder_conv_5")(encoder_activ_layer4)
    encoder_norm_layer5 = BatchNormalization(name="encoder_norm_5")(encoder_conv_layer5)
    encoder_activ_layer5 = ReLU()(encoder_norm_layer5)
    
    shape_before_flatten = K.int_shape(encoder_activ_layer5)[1:]
    encoder_flatten = Flatten()(encoder_activ_layer5)
    
    encoder_mu = Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
    encoder_log_variance =Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_flatten)
    
    encoder_mu_log_variance_model = Model(x, (encoder_mu, encoder_log_variance), name="encoder_mu_log_variance_model")
    
    def sampling(mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + K.exp(log_variance/2) * epsilon
        return random_sample
    
    encoder_output =Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])
    encoder_output=tanh(encoder_output)
    encoder =Model(x, encoder_output, name="encoder_model")
    
    decoder_input = Input(shape=(latent_space_dim), name="decoder_input")
    decoder_dense_layer1 = Dense(units=np.prod(shape_before_flatten), name="decoder_dense_1")(decoder_input)
    decoder_reshape = Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)
    decoder_conv_tran_layer1 =Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=1, name="decoder_conv_tran_1")(decoder_reshape)
    decoder_norm_layer1 = BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
    decoder_activ_layer1 =ReLU()(decoder_norm_layer1)
    
    decoder_conv_tran_layer2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="decoder_conv_tran_2")(decoder_activ_layer1)
    decoder_norm_layer2 = BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
    decoder_activ_layer2 = ReLU()(decoder_norm_layer2)
    
    decoder_conv_tran_layer3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="decoder_conv_tran_3")(decoder_activ_layer2)
    decoder_norm_layer3 = BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
    decoder_activ_layer3 = ReLU()(decoder_norm_layer3)
    
    decoder_conv_tran_layer4 =Conv2DTranspose(filters=1, kernel_size=(3, 3), padding="same", strides=1, name="decoder_conv_tran_4")(decoder_activ_layer3)
    decoder_output = ReLU()(decoder_conv_tran_layer4 )
    decoder = Model(decoder_input, decoder_output, name="decoder_model")
    
    vae_input = Input(shape=(y_size, x_size, num_channels), name="VAE_input")
    vae_encoder_output = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = Model(vae_input, vae_decoder_output, name="VAE")
    return vae, encoder, decoder, encoder_mu, encoder_log_variance