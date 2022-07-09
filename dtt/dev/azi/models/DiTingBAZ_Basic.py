import tensorflow as tf
from keras import Model, layers, optimizers, losses
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input, InputLayer, Lambda, Dense, Flatten, Conv1D, BatchNormalization, MaxPooling1D
from keras.layers import UpSampling1D, Cropping1D, Conv2DTranspose, Concatenate, Activation, add, Lambda, BatchNormalization, Dropout

def Conv1D_simple(x, t_channel_num,  activation_func='relu'):
    x_1 = layers.Conv1D(t_channel_num, 3, dilation_rate=1, activation=activation_func, padding='same')(x)
    return x_1

def ResBlock(x, t_channel_num, batch_norm = False, activation_func='relu'):
    x_in = x
    x_out = Conv1D_simple(x, t_channel_num, activation_func=None)
    if batch_norm:
        x = BatchNormalization()(x_out)
    x_out = Activation(activation_func)(x_out)
    x_out = Conv1D_simple(x_out, t_channel_num,  activation_func=None)
    if batch_norm:
        x = BatchNormalization()(x_out)
    x_out = Activation(activation_func)(x_out)
    x_out = add([x_in,x_out])

    return x_out

def Unet_downsampling_part(input_data, name=None):
    c_1_1 = Conv1D_simple(input_data, 8)
    c_1_2 = ResBlock(c_1_1, 8)

    p1 = MaxPooling1D(2,strides=2,padding='same')(c_1_2)
    c_2_1 = Conv1D_simple(p1, 16)
    c_2_2 = ResBlock(c_2_1, 16)

    p2 = MaxPooling1D(2,strides=2,padding='same')(c_2_2)
    c_3_1 = Conv1D_simple(p2, 32)
    c_3_2 = ResBlock(c_3_1, 32)

    p3 = MaxPooling1D(2,strides=2,padding='same')(c_3_2)
    c_4_1 = Conv1D_simple(p3, 64)
    c_4_2 = ResBlock(c_4_1, 64)

    p4 = MaxPooling1D(2,strides=2,padding='same')(c_4_2)
    c_5_1 = Conv1D_simple(p4, 128)
    c_5_2 = ResBlock(c_5_1, 128)

    return c_1_2, c_2_2, c_3_2, c_4_2, c_5_2

def DiTingBAZ_Basic(cfgs=None):
    """
    Stacked Unet
    """
    input_data = layers.Input(shape=(cfgs['Training']['Model']['length_before_P'] + cfgs['Training']['Model']['length_after_P'], cfgs['Training']['Model']['input_channel']),name='input')

    # stack 1 down part
    c_1_2, c_2_2, c_3_2, c_4_2, c_5_2 = Unet_downsampling_part(input_data, name='D_1')
    
    # stack 1 sin azi
    flatten_1_sin = Flatten()(c_5_2)
    dense_1_sin = Dense(32,activation='linear')(flatten_1_sin)
    dense_1_sin = Dense(8,activation='linear')(flatten_1_sin)
    dense_1_sin = Dense(1,activation='linear')(dense_1_sin)
    dense_1_sin_output = K.sin(dense_1_sin)
    # stack 1 cos azi
    flatten_1_cos = Flatten()(c_5_2)
    dense_1_cos = Dense(32,activation='linear')(flatten_1_cos)
    dense_1_cos = Dense(8,activation='linear')(flatten_1_cos)
    dense_1_cos = Dense(1,activation='linear')(dense_1_cos)
    dense_1_cos_output = K.cos(dense_1_cos)
    model = Model(inputs=input_data,outputs=[dense_1_sin_output, dense_1_cos_output])    

    if cfgs['Training']['Optimizer'] == 'adam':
        opt = optimizers.Adam(cfgs['Training']['LR'],clipvalue=5.0)
    else:
        pass

    model.compile(loss='mse',
                    optimizer=opt)
    
    model.summary()

    return model

if __name__ == '__main__':
    model = DiTingBAZ_Basic()
