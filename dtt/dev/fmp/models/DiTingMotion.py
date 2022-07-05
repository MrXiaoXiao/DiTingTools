import tensorflow as tf
from keras import Model, layers, optimizers, losses
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input, InputLayer, Lambda, Dense, Flatten, Conv1D, BatchNormalization, Dropout, MaxPooling1D
from keras.layers import UpSampling1D, Cropping1D, Concatenate, Activation, add, Lambda, Bidirectional, LSTM
from keras.layers import add, Activation, LSTM, Conv1D, Reshape
from keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization


def combo_conv(x, t_channel_num):
    x_1 = layers.Conv1D(t_channel_num, 3, dilation_rate=1, activation='relu', padding='same')(x)
    x_2 = layers.Conv1D(t_channel_num, 3, dilation_rate=2, activation='relu', padding='same')(x)
    x_3 = layers.Conv1D(t_channel_num, 5, dilation_rate=1, activation='relu', padding='same')(x)
    x_4 = layers.Conv1D(t_channel_num, 5, dilation_rate=2, activation='relu', padding='same')(x)
    
    x_out_concate = layers.Concatenate(axis=-1)([x_1, x_2, x_3, x_4, x])
    x_out_concate = Dropout(0.1)(x_out_concate)

    x_out  = layers.Conv1D(t_channel_num, 3, dilation_rate=1, activation='relu', padding='same')(x_out_concate)
    
    return x_out

def DiTingMotion(cfgs=None):
    """
    This function defines SmartMotion Model
    """
    # Input
    wave_input = layers.Input(shape=(cfgs['Training']['Model']['input_length'],cfgs['Training']['Model']['input_channel']), name='input')

    # Block 1
    x_b1_c1 = combo_conv(wave_input, 8)
    x_b1_c2 = combo_conv(x_b1_c1, 8)
    x_b1_concat = layers.Concatenate(axis=-1)([wave_input, x_b1_c2])
    x_p1 = layers.MaxPooling1D(2, strides=2, padding='same', name='block1_pool')(x_b1_concat)

    # Block 2
    x_b2_c1 = combo_conv(x_p1,8)
    x_b2_c2 = combo_conv(x_b2_c1,8)
    x_b2_concat = layers.Concatenate(axis=-1)([x_p1, x_b2_c2])
    x_p2 = layers.MaxPooling1D(2, strides=2, padding='same', name='block2_pool')(x_b2_concat)

    # Block 3
    x_b3_c1 = combo_conv(x_p2,8)
    x_b3_c2 = combo_conv(x_b3_c1,8)
    x_b3_c3 = combo_conv(x_b3_c2,8)
    x_b3_concat = layers.Concatenate(axis=-1)([x_p2, x_b3_c3])
    x_p3 = layers.MaxPooling1D(2, strides=2,  padding='same', name='block3_pool')(x_b3_concat)
    
    # Block 4
    x_b4_c1 = combo_conv(x_p3,8)
    x_b4_c2 = combo_conv(x_b4_c1,8)
    x_b4_c3 = combo_conv(x_b4_c2,8)
    x_b4_concat = layers.Concatenate(axis=-1)([x_p3, x_b4_c3])
    x_p4 = layers.MaxPooling1D(2, strides=2,  padding='same', name='block4_pool')(x_b4_concat)
    
    # Block 5
    x_b5_c1 = combo_conv(x_p4,8)
    x_b5_c2 = combo_conv(x_b5_c1,8)
    x_b5_c3 = combo_conv(x_b5_c2,8)
    x_b5_concat = layers.Concatenate(axis=-1)([x_p4, x_b5_c3])
    
    # motion
    x_p3_side = combo_conv(x_p3,2)
    x_p4_side = combo_conv(x_p4,2)
    x_p5_side = combo_conv(x_b5_concat,2)

    b3f = layers.Flatten()(x_p3_side)
    b4f = layers.Flatten()(x_p4_side)
    b5f = layers.Flatten()(x_p5_side)

    b3f_dense = layers.Dense(8,activation = 'relu')(b3f)
    b4f_dense = layers.Dense(8,activation = 'relu')(b4f)
    b5f_dense = layers.Dense(8,activation = 'relu')(b5f)

    # output
    o3  = layers.Dense(3,activation = 'sigmoid', name='o3')(b3f_dense)
    o4  = layers.Dense(3,activation = 'sigmoid', name='o4')(b4f_dense)
    o5  = layers.Dense(3,activation = 'sigmoid', name='o5')(b5f_dense)
    
    # fuse
    fuse = layers.Concatenate(axis=-1)([b3f_dense, b4f_dense, b5f_dense])
    fuse_2 = layers.Dense(8,activation = 'linear')(fuse)
    ofuse = layers.Dense(3,activation = 'sigmoid', name='ofuse')(fuse_2)

    # clarity
    x_p3_side_clarity = combo_conv(x_p3,2)
    x_p4_side_clarity = combo_conv(x_p4,2)
    x_p5_side_clarity = combo_conv(x_b5_concat,2)

    b3f_clarity = layers.Flatten()(x_p3_side_clarity)
    b4f_clarity = layers.Flatten()(x_p4_side_clarity)
    b5f_clarity = layers.Flatten()(x_p5_side_clarity)

    b3f_dense_clarity = layers.Dense(8,activation = 'relu')(b3f_clarity)
    b4f_dense_clarity = layers.Dense(8,activation = 'relu')(b4f_clarity)
    b5f_dense_clarity = layers.Dense(8,activation = 'relu')(b5f_clarity)

    # output
    o3_clarity  = layers.Dense(3,activation = 'sigmoid', name='o3_clarity')(b3f_dense_clarity)
    o4_clarity  = layers.Dense(3,activation = 'sigmoid', name='o4_clarity')(b4f_dense_clarity)
    o5_clarity  = layers.Dense(3,activation = 'sigmoid', name='o5_clarity')(b5f_dense_clarity)
    
    # fuse
    fuse_clarity = layers.Concatenate(axis=-1)([b3f_clarity, b4f_clarity, b5f_clarity])
    fuse_2_clarity = layers.Dense(8,activation = 'linear')(fuse_clarity)
    ofuse_clarity = layers.Dense(3,activation = 'sigmoid', name='ofuse_clarity')(fuse_2_clarity)
    
    output_dict = dict()
    
    output_dict['T0D0'] = o3
    output_dict['T0D1'] = o4
    output_dict['T0D2'] = o5
    output_dict['T0D3'] = ofuse

    output_dict['T1D0'] = o3_clarity
    output_dict['T1D1'] = o4_clarity
    output_dict['T1D2'] = o5_clarity
    output_dict['T1D3'] = ofuse_clarity
    
    # model
    model = Model(inputs=[wave_input], outputs = output_dict)
    model.summary()
    
    if cfgs['Training']['Optimizer'] == 'adam':
        opt = optimizers.Adam(cfgs['Training']['LR'])
    else:
        pass

    if cfgs['Training']['Loss'] == 'bce':
        model.compile(loss='binary_crossentropy',
                        metrics='accuracy',
                        optimizer=opt)

    return model