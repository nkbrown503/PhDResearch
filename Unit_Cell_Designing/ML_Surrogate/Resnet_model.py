# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:06:58 2022

@author: nbrow
"""


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow.math as tf_math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
def relu_advanced(x):
    return K.relu(x,max_value=20)
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    #bn = BatchNormalization()(relu)
    return relu

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(40, 120, 1))

    num_filters = 32
    #t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(inputs)
    t = relu_bn(t)
    
    num_blocks_list = [2, 2, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(pool_size=(2,2))(t)
    t = Flatten()(t)

    outputs = Dense(3,activation=relu_advanced)(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=['mean_squared_error'],
        metrics=['mean_squared_error']
    )

    return model