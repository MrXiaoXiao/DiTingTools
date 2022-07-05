import tensorflow as tf
from keras import Model, layers, optimizers
import keras.backend as K
import numpy as np
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Conv2DTranspose
from keras.layers import Concatenate, Activation, add, Lambda, BatchNormalization, Dropout

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation=None):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding,activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def Conv1D_simple(x, t_channel_num,  activation_func='relu'):
    x_out = layers.Conv1D(t_channel_num, 3, dilation_rate=1, activation=activation_func, padding='same')(x)
    return x_out

def ResBlock(x, t_channel_num, batch_norm = True, activation_func='relu'):
    x_in = x
    x_out = Conv1D_simple(x, t_channel_num, activation_func=None)
    if batch_norm:
        x = BatchNormalization()(x_out)
    x_out = Activation(activation_func)(x_out)
    x_out = Conv1D_simple(x_out, t_channel_num,  activation_func=None)
    x_out = Dropout(0.1)(x_out)
    if batch_norm:
        x = BatchNormalization()(x_out)
    x_out = Activation(activation_func)(x_out)
    x_out = add([x_in,x_out])

    return x_out

def Unet_downsampling_part(input_data, name=None):
    c_1_1 = Conv1D_simple(input_data, 8)
    c_1_2 = ResBlock(c_1_1, 8)

    p1 = MaxPooling1D(4,strides=4,padding='same')(c_1_2)
    c_2_1 = Conv1D_simple(p1, 16)
    c_2_2 = ResBlock(c_2_1, 16)

    p2 = MaxPooling1D(4,strides=4,padding='same')(c_2_2)
    c_3_1 = Conv1D_simple(p2, 32)
    c_3_2 = ResBlock(c_3_1, 32)

    p3 = MaxPooling1D(4,strides=4,padding='same')(c_3_2)
    c_4_1 = Conv1D_simple(p3, 64)
    c_4_2 = ResBlock(c_4_1, 64)

    p4 = MaxPooling1D(4,strides=4,padding='same')(c_4_2)
    c_5_1 = Conv1D_simple(p4, 128)
    c_5_2 = ResBlock(c_5_1, 128)

    return c_1_2, c_2_2, c_3_2, c_4_2, c_5_2

def Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2, name=None):
    
    u5 = Conv1DTranspose(c_5_2, filters=64,kernel_size=8,strides=4,padding='same')
    concate_4 = Concatenate(axis=-1)([u5, c_4_2])
    concate_4_conv = Conv1D_simple(concate_4, 64)
    concate_4_conv = ResBlock(concate_4_conv, 64)

    u4 = Conv1DTranspose(concate_4_conv, filters=32,kernel_size=8,strides=4,padding='same')
    concate_3 = Concatenate(axis=-1)([u4, c_3_2])
    concate_3_conv = Conv1D_simple(concate_3, 32)
    concate_3_conv = ResBlock(concate_3_conv, 32)
    
    u3 = Conv1DTranspose(concate_3_conv, filters=16,kernel_size=8,strides=4,padding='same')
    concate_2 = Concatenate(axis=-1)([u3, c_2_2])
    concate_2_conv = Conv1D_simple(concate_2, 16)
    concate_2_conv = ResBlock(concate_2_conv, 16)

    u2 = Conv1DTranspose(concate_2_conv, filters=8,kernel_size=8,strides=4,padding='same')
    concate_1 = Concatenate(axis=-1)([u2, c_1_2])
    concate_1_conv = Conv1D_simple(concate_1, 8)
    concate_1_conv = ResBlock(concate_1_conv, 8)

    return concate_1_conv

def DiTingPicker(cfgs=None):
    """
    function for returning DiTingPicker model.
    input: cfgs. yaml configuration data.
    output: compiled deep learning model.
    """
    input_data = layers.Input(shape=(cfgs['Training']['Model']['input_length'],cfgs['Training']['Model']['input_channel']),name='input')

    # stack 1 down part
    c_1_2, c_2_2, c_3_2, c_4_2, c_5_2 = Unet_downsampling_part(input_data, name='D_1')
    # stack 1 p
    stack_1_p = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2, name='U_P_1')
    pred_1_p = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_1_p')(stack_1_p)
    # stack 1 s
    stack_1_s = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2, name='U_S_1')
    pred_1_s = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_1_s')(stack_1_s)
    # stack 1 d
    stack_1_d = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2, name='U_D_1')
    pred_1_d = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_1_d')(stack_1_d)

    # stack 2 down part
    input_data_stack = Concatenate(axis=-1)([input_data, stack_1_p, stack_1_s, stack_1_d])
    c_1_2, c_2_2, c_3_2, c_4_2, c_5_2 = Unet_downsampling_part(input_data_stack,name='D_2')
    # stack 2 p
    stack_2_p = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2,name='U_P_2')
    pred_2_p = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_2_p')(stack_2_p)
    # stack 2 s
    stack_2_s = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2, name='U_S_2')
    pred_2_s = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_2_s')(stack_2_s)
    # stack 2 d
    stack_2_d = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2,name='U_D_2')
    pred_2_d = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_2_d')(stack_2_d)
        
    # stack 3 down part
    input_data_stack = Concatenate(axis=-1)([input_data, stack_1_p, stack_1_s, stack_2_p, stack_2_s, stack_1_d, stack_2_d])
    c_1_2, c_2_2, c_3_2, c_4_2, c_5_2 = Unet_downsampling_part(input_data_stack,name='D_3')
    # stack 3 p
    stack_3_p = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2,name='U_P_3')
    pred_3_p = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_3_p')(stack_3_p)
    # stack 3 s
    stack_3_s = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2,name='U_S_3')
    pred_3_s = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_3_s')(stack_3_s)
    # stack 2 d
    stack_3_d = Unet_upsampling_part(c_1_2, c_2_2, c_3_2, c_4_2, c_5_2,name='U_D_3')
    pred_3_d = Conv1D(1,3,padding='same',activation='sigmoid',name='stack_3_d')(stack_3_d)
    
    final_p = Concatenate(axis=-1)([stack_1_p, stack_2_p, stack_3_p])
    final_p = Conv1D(1,3,padding='same',activation='sigmoid',name='final_p')(final_p)

    final_s = Concatenate(axis=-1)([stack_1_s, stack_2_s, stack_3_s])
    final_s = Conv1D(1,3,padding='same',activation='sigmoid',name='final_s')(final_s)
    
    final_d = Concatenate(axis=-1)([stack_1_d, stack_2_d, stack_3_d])
    final_d = Conv1D(1,3,padding='same',activation='sigmoid',name='final_d')(final_d)

    output_dict = dict()
    
    output_dict['C0D0'] = pred_1_p
    output_dict['C0D1'] = pred_2_p
    output_dict['C0D2'] = pred_3_p
    output_dict['C0D3'] = final_p

    output_dict['C1D0'] = pred_1_s
    output_dict['C1D1'] = pred_2_s
    output_dict['C1D2'] = pred_3_s
    output_dict['C1D3'] = final_s
    
    output_dict['C2D0'] = pred_1_d
    output_dict['C2D1'] = pred_2_d
    output_dict['C2D2'] = pred_3_d
    output_dict['C2D3'] = final_d
    
    model = Model(inputs=input_data,outputs=output_dict)  


    if cfgs['Training']['Optimizer'] == 'adam':
        opt = optimizers.Adam(cfgs['Training']['LR'], clipvalue=cfgs['Training']['clipvalue'])
    else:
        pass
    
    if cfgs['Training']['Loss'] == 'bce':
        model.compile(loss='binary_crossentropy',optimizer=opt)
    else:
        print('Loss: {}'.format(cfgs['Training']['Loss']))
        print('Model not complied!!!')
    if cfgs['Training']['Model']['show_summary']:
        model.summary()
    return model

if __name__ == '__main__':
    model = DiTingPicker()