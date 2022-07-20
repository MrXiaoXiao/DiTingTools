import h5py
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from random import shuffle
import matplotlib.pyplot as plt
from numpy.fft import irfft, rfftfreq
from numpy import sqrt, newaxis
from numpy.random import normal
from scipy import signal
import obspy

####################################################
# Functions for reading datasets with a given key
####################################################

def get_from_DiTing(part=None, key=None, h5file_path=''):
    """
    get waveform from DiTing Dataset
    TODO: description
    """
    with h5py.File(h5file_path + 'DiTing330km_part_{}.hdf5'.format(part), 'r') as f:
        dataset = f.get('earthquake/'+str(key))    
        data = np.array(dataset).astype(np.float32)
        up_sample_data = np.zeros([data.shape[0]*2, data.shape[1]])
        for chdx in range(data.shape[1]):
            up_sample_data[:,chdx] = signal.resample(data[:,chdx], len(data[:,chdx]) * 2)
        data = up_sample_data
    return data

def get_from_STEAD(key=None, h5file_path='', is_noise=False):
    """
    get waveform from STEAD Dataset
    TODO: description
    """
    with h5py.File(h5file_path, 'r') as f:
        if is_noise:
            dataset = f.get('non_earthquake/noise/'+str(key))
            data = np.array(dataset).astype(np.float32)
        else:
            dataset = f.get('earthquake/local/'+str(key))
            data = np.array(dataset).astype(np.float32)
    return data[:,::-1]

def get_from_INSTANCE(key=None, h5file_path='', is_noise=False):
    """
    get waveform from INSTANCE Dataset
    TODO: description
    """
    with h5py.File(h5file_path, 'r') as f:
        dataset = f.get('data/'+str(key))
        data_t = np.array(dataset).astype(np.float32)
        data = np.zeros([12000,3])
        data[:,0] = data_t[2,:]
        data[:,1] = data_t[1,:]
        data[:,2] = data_t[0,:]
    return data

def get_from_SCSN_FMP(index, h5file_path=''):
    """
    get waveform from SCSN FMP
    TODO: description
    """
    with h5py.File(h5file_path, 'r') as f:
        data = np.asarray(f['X'][index])
        label = np.asarray(f['Y'][index])
    return data, label

def get_from_SCSN_P(index, h5file_path=''):
    """
    get waveform from SCSN P
    TODO: description
    """
    with h5py.File(h5file_path, 'r') as f:
        data = np.asarray(f['X'][index])
    return data

def gen_colored_noise(alpha, length, dt = 0.01, fmin=0):
    # calculate freq
    f = rfftfreq(length, dt)
    # scaling factor
    s_scale = f
    fmin = max(fmin, 1./(length*dt))
    cutoff_index   = np.sum(s_scale < fmin)
    if cutoff_index and cutoff_index < len(s_scale):
        s_scale[:cutoff_index] = s_scale[cutoff_index]
    s_scale = s_scale**(-alpha/2.)
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (length % 2)) / 2.
    sigma = 2 * sqrt(np.sum(w**2)) / length
    size = [len(f)]
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)
    if not (length % 2): si[...,-1] = 0
    si[...,0] = 0
    s  = sr + 1J * si
    y = irfft(s, n=length, axis=-1) / sigma
    return y

def get_periodic_energy(noise_length):
    noise_data = np.zeros([noise_length, 3])
    shift_choice = np.random.choice([-1,1])
    disp_gap = np.random.uniform(low=0.0008, high=0.008)
    if shift_choice == 1:
        shift_index = np.random.randint(low=0, high=noise_length)
        for idx in range(3):
            energy_distrub = np.cos(np.arange(0,disp_gap*noise_length,disp_gap)[:noise_length] + np.random.uniform(low=-disp_gap*3,high=disp_gap*3,size=noise_length))*np.random.uniform(low=-0.1,high=0.1,size=noise_length)
            noise_data[shift_index:, idx] = energy_distrub[:noise_length - shift_index]
    else:
        for idx in range(3):
            noise_data[:, idx] = np.cos(np.arange(0,disp_gap*noise_length,disp_gap)[:noise_length] + np.random.uniform(low=-disp_gap*3,high=disp_gap*3,size=noise_length))*np.random.uniform(low=-0.1,high=0.1,size=noise_length)
    return noise_data

def get_long_rect_misguide(noise_length):
    noise_data = np.zeros([noise_length, 3])
    for idx in range(3):
        boundary_rate = np.random.uniform(low=0.1, high= 0.1)
        top_value = np.random.uniform(low=0.01,high=100.0)
        bottom_value = np.random.uniform(low=-100.0,high=-0.1)
        bound_dx = int(noise_length*boundary_rate)
        noise_data[bound_dx:(noise_length-bound_dx),idx] = top_value
        noise_data[:bound_dx,idx] = bottom_value
        noise_data[(noise_length-bound_dx):,idx] = bottom_value
    reverse_factor = np.random.choice([-1,1])
    noise_data *= reverse_factor
    return noise_data

def get_disalign_misguide(noise_length):
    noise_data = np.zeros([noise_length, 3])
    shift_index = np.random.randint(low=0, high=int(noise_length*0.8))
    alpha = np.random.uniform(low=0,high=4)
    taper_value = np.random.uniform(low=0, high=0.1)
    
    # signify_width = np.random.randint(low=1, high=int(noise_length*0.09))
    # signify_scale = np.random.uniform(low=1.0,high=10.0)
    
    for idx in range(3):
        temp_data = gen_colored_noise(alpha, noise_length)
        temp_data_pad_zeros = np.zeros(noise_length)
        if idx != 0:
            disalign_shift = np.random.randint(low=0,high=int(noise_length*0.09))
        else:
            disalign_shift = 0
        
        temp_data_pad_zeros[shift_index+disalign_shift:] = temp_data[:noise_length - shift_index - disalign_shift]
        temp_trace = obspy.Trace(data=temp_data_pad_zeros)
        temp_trace.stats.sampling_rate = 100.0
        filter_choice = np.random.choice([0,1])
        
        if filter_choice == 1:
            temp_trace.filter('bandpass',freqmin=0.5,freqmax=40,zerophase=True)
        elif filter_choice == 0:
            temp_trace.filter('bandpass',freqmin=np.random.uniform(low=0.1,high=2.0),freqmax=np.random.uniform(low=8.0,high=45.0),zerophase=True)
        
        temper_choice = np.random.choice([-1,1])

        if temper_choice == 1:
            temp_trace.taper(0.05)
        else:
            temp_trace.taper(taper_value)

        noise_data[:, idx] = temp_trace.data[:]   
    return noise_data
    
def get_artifical_boundary(noise_length):
    noise_data = np.zeros([noise_length, 3])
    shift_index = np.random.randint(low=0, high=noise_length)
    alpha = np.random.uniform(low=0,high=2)
    taper_value = np.random.uniform(low=0, high=0.1)
    shift_type_choice = np.random.choice([-1,1])
    if shift_type_choice == 1:
        for idx in range(3):
            temp_data = gen_colored_noise(alpha, noise_length)
            temp_data_pad_zeros = np.zeros(noise_length)
            temp_data_pad_zeros[shift_index:] = temp_data[:noise_length - shift_index]
            temp_trace = obspy.Trace(data=temp_data_pad_zeros)
            temp_trace.stats.sampling_rate = 100.0
            filter_choice = np.random.choice([-1,0,1])
            if filter_choice == 1:
                temp_trace.filter('bandpass',freqmin=0.5,freqmax=40,zerophase=True)
            elif filter_choice == -1:
                pass
            elif filter_choice == 0:
                temp_trace.filter('bandpass',freqmin=np.random.uniform(low=0.1,high=2.0),freqmax=np.random.uniform(low=8.0,high=45.0),zerophase=True)
            temper_choice = np.random.choice([-1,1])

            if temper_choice == 1:
                temp_trace.taper(0.05)
            else:
                temp_trace.taper(taper_value)

            noise_data[:, idx] = temp_trace.data[:]
    elif shift_type_choice == -1:
        for idx in range(3):
            temp_data = gen_colored_noise(alpha, noise_length)
            temp_data_pad_zeros = np.zeros(noise_length)
            temp_data_pad_zeros[:noise_length - shift_index] = temp_data[:noise_length - shift_index]
            temp_trace = obspy.Trace(data=temp_data_pad_zeros)
            temp_trace.stats.sampling_rate = 100.0
            filter_choice = np.random.choice([-1,0,1])
            if filter_choice == 1:
                temp_trace.filter('bandpass',freqmin=0.5,freqmax=40,zerophase=True)
            elif filter_choice == -1:
                pass
            elif filter_choice == 0:
                temp_trace.filter('bandpass',freqmin=np.random.uniform(low=0.1,high=2.0),freqmax=np.random.uniform(low=8.0,high=45.0),zerophase=True)
            temper_choice = np.random.choice([-1,1])

            if temper_choice == 1:
                temp_trace.taper(0.05)
            else:
                temp_trace.taper(taper_value)

            noise_data[:, idx] = temp_trace.data[:]
        
    return noise_data

def get_simple_misguide(noise_length, misguide_width_min = 10, misguide_width_max = 300):
    misguide_P_pos = np.random.randint(low=0,high=noise_length - misguide_width_max*8)
    misguide_S_pos = np.random.randint(low=misguide_P_pos + 10,high=noise_length - misguide_width_max*4)
    misguide_P_decay = np.random.uniform(low=0.1,high=2.0)
    misguide_S_decay = np.random.uniform(low=0.1,high=2.0)
    
    #available_misguide_type = ['sine', 'triangle', 'spike', 'rect']
    available_misguide_type = ['sine', 'triangle', 'spike', 'rect', 'ricker', 'random', 'empty']
    misguide_type = np.random.choice(available_misguide_type)
    misguide_length = np.random.randint(low=misguide_width_min, high=misguide_width_max)
    
    if misguide_type == 'sine':
        sine_value = np.sin(np.arange(0,2*np.pi,2*np.pi/misguide_length))
        sine_value_len = len(sine_value)
        misguide_data = np.zeros([noise_length, 3])
        misguide_data[misguide_P_pos:misguide_P_pos+sine_value_len, 0] = sine_value[:]
        misguide_data[misguide_P_pos:misguide_P_pos+sine_value_len, 1] = sine_value[:]*misguide_S_decay
        misguide_data[misguide_P_pos:misguide_P_pos+sine_value_len, 2] = sine_value[:]*misguide_S_decay
        
        misguide_data[misguide_S_pos:misguide_S_pos+sine_value_len, 0] = sine_value[:]*misguide_P_decay
        misguide_data[misguide_S_pos:misguide_S_pos+sine_value_len, 1] = sine_value[:]
        misguide_data[misguide_S_pos:misguide_S_pos+sine_value_len, 2] = sine_value[:]
    
    elif misguide_type == 'triangle':
        tri_time = np.arange(-0.5,0.5,1/misguide_length)
        tri_value = np.abs(signal.sawtooth(2 * np.pi* tri_time))*np.random.choice([-1,1])
        tri_value_len = len(tri_value)
        misguide_data = np.zeros([noise_length, 3])
        misguide_data[misguide_P_pos:misguide_P_pos+tri_value_len, 0] = tri_value[:]
        misguide_data[misguide_P_pos:misguide_P_pos+tri_value_len, 1] = tri_value[:]*misguide_S_decay
        misguide_data[misguide_P_pos:misguide_P_pos+tri_value_len, 2] = tri_value[:]*misguide_S_decay
        
        misguide_data[misguide_S_pos:misguide_S_pos+tri_value_len, 0] = tri_value[:]*misguide_P_decay
        misguide_data[misguide_S_pos:misguide_S_pos+tri_value_len, 1] = tri_value[:]
        misguide_data[misguide_S_pos:misguide_S_pos+tri_value_len, 2] = tri_value[:]
    
    elif misguide_type == 'spike':
        misguide_data = np.zeros([noise_length, 3])
        misguide_data[misguide_P_pos, 0] = 1.0*np.random.choice([-1,1])
        misguide_data[misguide_P_pos, 1] = 1.0*misguide_S_decay*np.random.choice([-1,1])
        misguide_data[misguide_P_pos, 2] = 1.0*misguide_S_decay*np.random.choice([-1,1])
        
        misguide_data[misguide_S_pos, 0] = 1.0*misguide_P_decay*np.random.choice([-1,1])
        misguide_data[misguide_S_pos, 1] = 1.0*np.random.choice([-1,1])
        misguide_data[misguide_S_pos, 2] = 1.0*np.random.choice([-1,1])
    
    elif misguide_type == 'rect':
        misguide_data = np.zeros([noise_length, 3])
        rect_flag = np.random.choice([-1,1])
        misguide_data[misguide_P_pos:misguide_P_pos+misguide_length, 0] = rect_flag
        misguide_data[misguide_P_pos:misguide_P_pos+misguide_length, 1] = rect_flag*misguide_S_decay
        misguide_data[misguide_P_pos:misguide_P_pos+misguide_length, 2] = rect_flag*misguide_S_decay
        
        misguide_data[misguide_S_pos:misguide_S_pos+misguide_length, 0] = rect_flag*misguide_P_decay
        misguide_data[misguide_S_pos:misguide_S_pos+misguide_length, 1] = rect_flag
        misguide_data[misguide_S_pos:misguide_S_pos+misguide_length, 2] = rect_flag
        
    elif misguide_type == 'ricker':
        misguide_data = np.zeros([noise_length, 3])
        ricker_value = signal.ricker(misguide_length, 16)*np.random.choice([-1,1])
        ricker_value_len = len(ricker_value)
        misguide_data[misguide_P_pos:misguide_P_pos+ricker_value_len, 0] = ricker_value[:]
        misguide_data[misguide_P_pos:misguide_P_pos+ricker_value_len, 1] = ricker_value[:]*misguide_S_decay
        misguide_data[misguide_P_pos:misguide_P_pos+ricker_value_len, 2] = ricker_value[:]*misguide_S_decay
        
        misguide_data[misguide_S_pos:misguide_S_pos+ricker_value_len, 0] = ricker_value[:]*misguide_P_decay
        misguide_data[misguide_S_pos:misguide_S_pos+ricker_value_len, 1] = ricker_value[:]
        misguide_data[misguide_S_pos:misguide_S_pos+ricker_value_len, 2] = ricker_value[:]
    
    elif misguide_type == 'random':
        misguide_data = np.zeros([noise_length, 3])
        random_value = np.random.normal(0.0, 1.0, size = misguide_length)*np.random.choice([-1,1])
        random_value_len = len(random_value)
        misguide_data[misguide_P_pos:misguide_P_pos+random_value_len, 0] = random_value[:]
        misguide_data[misguide_P_pos:misguide_P_pos+random_value_len, 1] = random_value[:]*misguide_S_decay
        misguide_data[misguide_P_pos:misguide_P_pos+random_value_len, 2] = random_value[:]*misguide_S_decay
        
        misguide_data[misguide_S_pos:misguide_S_pos+random_value_len, 0] = random_value[:]*misguide_P_decay
        misguide_data[misguide_S_pos:misguide_S_pos+random_value_len, 1] = random_value[:]
        misguide_data[misguide_S_pos:misguide_S_pos+random_value_len, 2] = random_value[:]
    
    elif misguide_type == 'empty':
        misguide_data = np.zeros([noise_length, 3])
    
    available_noise_type = ['colorful', 'none']
    noise_type = np.random.choice(available_noise_type)
    if noise_type == 'none':
        pass
    elif noise_type == 'colorful':
        for idx in range(3):
            alpha = np.random.uniform(low=0,high=2)
            noise_data = gen_colored_noise(alpha, noise_length)
            noise_data /= np.max(np.abs(noise_data))
            noise_factor = np.random.uniform(low=0,high=0.3)
            misguide_data[:,idx] += noise_data*noise_factor
    
    return misguide_data

####################################################
# Functions for creating a training instance
####################################################
def get_shifted_instance_for_EqDetPhasePicking_training(dataset_name = 'DiTing',
                                                        dataset_path = None,
                                                        data_length = 6144,
                                                        part = None,
                                                        key = None,
                                                        P = None,
                                                        S = None,
                                                        label_length = 101):
    temp_data_X = np.zeros([int(data_length),3])
    temp_data_Y = np.zeros([int(data_length),3])
    
    # fetch data
    if dataset_name == 'DiTing':
        try:
            data = get_from_DiTing(part=part, key=key, h5file_path=dataset_path)
        except:
            print('Error on key {} DiTing'.format(key))
            return temp_data_X, temp_data_Y
    elif dataset_name == 'STEAD':
        try:
            data = get_from_STEAD(key=key, h5file_path=dataset_path)
        except:
            print('Error on key {} STEAD'.format(key))
            return temp_data_X, temp_data_Y
    elif dataset_name == 'INSTANCE':
        try:
            data = get_from_INSTANCE(key=key, h5file_path=dataset_path)
        except:
            print('Error on key {} INSTANCE'.format(key))
            return temp_data_X, temp_data_Y
    else:
        print('Dataset Type Not Supported!!!')
        return

    # augmentation
    # shift data
    ori_length = np.shape(data)[0]
    if ori_length < data_length:
        temp_data_X[:ori_length,:] = data[:ori_length,:]
        origin_shift = 0
    else:
        if S >= data_length:
            origin_shift = P - np.random.randint(200, 1000)
        else:
            origin_shift = 0
        temp_data_X[:data_length,:] = data[origin_shift:origin_shift + data_length,:]
        
    shift_sample = np.random.randint(low=(P-origin_shift)*(-1) + 1, high=data_length - S + origin_shift - 2)
    temp_data_X = np.roll(temp_data_X, shift_sample, axis=0)
    # STD normalization here
    for chdx in range(3):
        temp_data_X[:,chdx] -= np.mean(temp_data_X[:,chdx])
        norm_factor = np.std(temp_data_X[:,chdx])
        if norm_factor == 0:
            pass
        else:
            temp_data_X[:,chdx] /= norm_factor
    # create label
    label_P, label_S, label_D = label_EqDetPick(P - origin_shift, S - origin_shift, data_length, shift = shift_sample, label_length = label_length)
    temp_data_Y[:,0] = label_P[:]
    temp_data_Y[:,1] = label_S[:]
    temp_data_Y[:,2] = label_D[:]    

    return temp_data_X, temp_data_Y

def get_augmented_instance_for_EqDetPhasePicking_training(dataset_name='DiTing',
                                                            dataset_path = None,
                                                            data_length = 6144,
                                                            temp_part_list = None,
                                                            key_list = None,
                                                            P_list = None,
                                                            S_list = None,
                                                            pad_noise_prob = 0.2,
                                                            min_noise_len = 200,
                                                            max_noise_len = 1000,
                                                            max_time = 2,
                                                            label_length = 101
                                                            ):
    
    temp_data_X = np.zeros([int(data_length*max_time),3])
    temp_data_Y = np.zeros([int(data_length*max_time),3])

    start_index = 0

    for key_dx, key in enumerate(key_list):
        if np.random.uniform(low=0,high=1) <= pad_noise_prob:
            noise_len = np.random.randint(low=min_noise_len, high=max_noise_len)
            if noise_len + start_index >= data_length*max_time:
                noise_len =  data_length*max_time - start_index

            noise_type_prob = np.random.uniform(low=0,high=1)
            if noise_type_prob <= 0.3:
                temp_data_X[start_index:start_index+noise_len,:] = np.random.normal(loc=0.0,scale=2.0,size=np.shape(temp_data_X[start_index:start_index+noise_len,:]))
                temp_data_Y[start_index:start_index+noise_len,:] = 0
            elif noise_type_prob > 0.3 and noise_type_prob < 0.5:
                temp_data_X[start_index:start_index+noise_len,:] = 0
                temp_data_X[start_index+int(noise_len*0.3):start_index+int(noise_len*0.7),:] = 1.0
                temp_data_Y[start_index:start_index+noise_len,:] = 0
            elif noise_type_prob >= 0.5 and noise_type_prob < 0.8:
                temp_data_X[start_index:start_index+noise_len,:] = np.random.uniform(low=-1.5,high=1.5,size=np.shape(temp_data_X[start_index:start_index+noise_len,:]))
                temp_data_Y[start_index:start_index+noise_len,:] = 0
            else:
                alpha = np.random.uniform(low=0,high=2)
                if noise_len < 200:
                    temp_data_X[start_index:start_index+noise_len,:] = 0
                else:    
                    for noise_channel_dx in range(3):
                        t_noise = gen_colored_noise(alpha=alpha, length=noise_len, dt=0.01, fmin=0)
                        t_noise /= np.std(t_noise)
                        temp_data_X[start_index:start_index+noise_len,noise_channel_dx] = t_noise[:]

                temp_data_Y[start_index:start_index+noise_len,:] = 0                  
            start_index += noise_len

        if start_index >= data_length*max_time:
            break

        p_t = int(P_list[key_dx])
        s_t = int(S_list[key_dx])    

        if dataset_name == 'DiTing':
            try:
                data = get_from_DiTing(part=temp_part_list[key_dx], key=key, h5file_path=dataset_path)
            except:
                print('Error on key {} DiTing'.format(key))
                continue

            # shift and crop data
            p_shift = np.random.randint(high=3000,low=500)
            s_shift = np.random.randint(high=int(6*(s_t-p_t)),low=int(2*(s_t-p_t)))
            if p_t - p_shift <= 0:
                p_shift = p_t - 100
            if s_t + s_shift >= len(data):
                s_shift = len(data) - s_t - 200
        
        elif dataset_name == 'STEAD':
            try:
                data = get_from_STEAD(key=key, h5file_path=dataset_path)
            except:
                print('Error on key {} STEAD'.format(key))
                continue
            
            p_shift = np.random.randint(high=2000,low=500)
            s_shift = np.random.randint(high=int(10*(s_t-p_t)),low=int(2*(s_t-p_t)))
            if p_t - p_shift <= 0:
                p_shift = p_t - 150
            if s_t + s_shift >= len(data):
                s_shift = len(data) - s_t - 200
        
        elif dataset_name == 'INSTANCE':
            try:
                data = get_from_INSTANCE(key=key, h5file_path=dataset_path)
            except:
                print('Error on key {} INSTANCE'.format(key))
                continue

            p_shift = np.random.randint(high=2000,low=500)
            s_shift = np.random.randint(high=int(8*(s_t-p_t)),low=int(2*(s_t-p_t)))
            if p_t - p_shift <= 0:
                p_shift = p_t - 100
            if s_t + s_shift >= len(data):
                s_shift = len(data) - s_t - 200
        else:
            print('Dataset Type Not Supported!!!')
            return
        
        data = data[p_t - p_shift: s_t + s_shift, :]
        pad_len = int((s_t+s_shift - p_t + p_shift) - 1)
                
        if start_index + pad_len > data_length*max_time:
            pad_len = data_length*max_time - start_index

        label_P, label_S, label_D = label_EqDetPick(p_shift, p_shift + s_t - p_t, data_length= len(data), label_length=label_length)

        reverse_factor = np.random.choice([-1,1])
        rescale_factor = np.random.uniform(low=0.5,high=1.5)

        for chn_dx in range(3):
            data[:,chn_dx] -= np.mean(data[:,chn_dx])
            norm_factor = np.std(data[:,chn_dx])
            if norm_factor == 0:
                pass
            else:
                data[:,chn_dx] /= norm_factor
            
            data[:,chn_dx] *= rescale_factor
            data[:,chn_dx] *= reverse_factor

            if dataset_name == 'DiTing':
                temp_data_X[start_index:start_index + pad_len,chn_dx] = data[:pad_len,chn_dx]
            if dataset_name == 'STEAD' or dataset_name == 'INSTANCE':
                if chn_dx == 0:
                    temp_data_X[start_index:start_index + pad_len,2] = data[:pad_len,chn_dx]
                elif chn_dx == 1:
                    temp_data_X[start_index:start_index + pad_len,1] = data[:pad_len,chn_dx]
                elif chn_dx == 2:
                    temp_data_X[start_index:start_index + pad_len,0] = data[:pad_len,chn_dx]
        
        temp_data_Y[start_index:start_index + pad_len,0] = label_P[:pad_len]
        temp_data_Y[start_index:start_index + pad_len,1] = label_S[:pad_len]
        temp_data_Y[start_index:start_index + pad_len,2] = label_D[:pad_len]
        start_index += pad_len
    
    shift_dx = np.where(temp_data_Y[:data_length*int(max_time-1),2]==0)[0]
    shift_dx = np.random.choice(shift_dx)
    temp_data_X = temp_data_X[shift_dx:shift_dx+data_length,:]

    add_noise_prob = np.random.uniform(low=0,high=1.0)
    noise_level_ratio = np.random.uniform(low=0.001,high=0.010)

    # STD normalization here
    for chdx in range(3):
        if add_noise_prob > 0.8:
            temp_data_X[:,chdx] += np.random.uniform(low=noise_level_ratio*np.min(temp_data_X[:,chdx]),
                                                    high=noise_level_ratio*np.max(temp_data_X[:,chdx]),
                                                    size=np.shape(temp_data_X[:,chdx]))

        temp_data_X[:,chdx] -= np.mean(temp_data_X[:,chdx])
        norm_factor = np.std(temp_data_X[:,chdx])
        if norm_factor == 0:
            pass
        else:
            temp_data_X[:,chdx] /= norm_factor
    
    temp_data_Y = temp_data_Y[shift_dx:shift_dx+data_length,:]
    
    return temp_data_X, temp_data_Y

def get_DiTing_EDPP_Negtive_example(noise_length):
    available_noise_type = ['simple_misguide', 'artifical_boundary', 'periodic_energy', 'disalign_misguide', 'long_rect_misguide']
    noise_type = np.random.choice(available_noise_type)
    if noise_type == 'simple_misguide':
        noise_data = get_simple_misguide(noise_length)
    elif noise_type == 'artifical_boundary':
        noise_data = get_artifical_boundary(noise_length)
    elif noise_type == 'periodic_energy':
        noise_data = get_periodic_energy(noise_length)
    elif noise_type == 'disalign_misguide':
        noise_data = get_disalign_misguide(noise_length)
    elif noise_type == 'long_rect_misguide':
        noise_data = get_long_rect_misguide(noise_length)

    for chdx in range(3):
        noise_data[:,chdx] -= np.mean(noise_data[:,chdx])
        norm_factor = np.std(noise_data[:,chdx])
        if norm_factor == 0:
            pass
        else:
            noise_data[:,chdx] /= norm_factor
    noise_Y = np.zeros([noise_length, 3])
    return noise_data, noise_Y

def get_real_noise_for_EqDetPhasePicking_training(dataset_name='STEAD_NO',
                                                dataset_path = None,
                                                data_length = 6144,
                                                key_list = None,
                                                max_time = 2
                                                ):
    
    temp_data_X = np.zeros([int(data_length*max_time),3])
    temp_data_Y = np.zeros([int(data_length),3])

    start_index = 0

    for key in key_list:
        if start_index >= data_length*max_time:
            temp_data_X[:,chdx] -= np.mean(temp_data_X[:,chdx])
                    
        if dataset_name == 'TLSC_NO':
            pass # private dataset
        elif dataset_name == 'STEAD_NO':
            data = get_from_STEAD(key=key, h5file_path=dataset_path, is_noise=True)
        elif dataset_name == 'INSTANCE_NO':
            data = get_from_INSTANCE(key=key, h5file_path=dataset_path)
        else:
            print('Unsupported noise set: {}'.format(dataset_name))
        
        temp_length = np.shape(data)[0]
        
        for chdx in range(3):
            data[:,chdx] -= np.mean(data[:,chdx])
            norm_factor = np.std(data[:,chdx])
            if norm_factor == 0:
                pass
            else:
                data[:,chdx] /= np.std(data[:,chdx])
        
        if start_index + temp_length >= data_length*max_time:
            append_length = data_length*max_time - start_index
            temp_data_X[start_index:, :] = data[:append_length,:]
            start_index += append_length
        else:
            append_length = temp_length
            temp_data_X[start_index:start_index+append_length, :] = data[:,:]
            start_index += append_length
    
    shift_dx = np.random.randint( low=0,high=data_length*(max_time-1) )
    temp_data_X = temp_data_X[shift_dx:shift_dx+data_length,:]

    # STD normalization here
    for chdx in range(3):
        temp_data_X[:,chdx] -= np.mean(temp_data_X[:,chdx])
        norm_factor = np.std(temp_data_X[:,chdx])
        if norm_factor == 0:
            pass
        else:
            temp_data_X[:,chdx] /= norm_factor

    return temp_data_X, temp_data_Y

####################################################
# Functions for creating a training generator
####################################################
class DiTingGenerator:
    # init function
    def __init__(self, csv_hdf5_mapping_dict):
        self.csv_file = pd.read_csv(csv_hdf5_mapping_dict['csv_path'], dtype = {'key': str})
        self.name = csv_hdf5_mapping_dict['name']
        
        if self.name not in ['DiTing', 'STEAD', 'INSTANCE']:
            print('Dataset type not Supported Yet!!!')
        
        if self.name == 'DiTing':
            self.key_str = 'key'
            self.part_str = 'part'
            self.p_str = 'p_pick'
            self.s_str = 's_pick'
        elif self.name == 'STEAD':
            self.key_str = 'trace_name'
            self.p_str = 'p_arrival_sample'
            self.s_str = 's_arrival_sample'
        elif self.name == 'INSTANCE':
            self.key_str = 'trace_name'
            self.p_str = 'trace_P_arrival_sample'
            self.s_str = 'trace_S_arrival_sample'
        
        self.has_parts = csv_hdf5_mapping_dict['has_parts']
        """
        if self.has_parts:
            self.part_num = csv_hdf5_mapping_dict['part_num']
            self.part_list = []
            for idx in range(self.part_num):
                self.part_list.append( h5py.File(csv_hdf5_mapping_dict['part_list'][idx], 'r') )
        else:
            self.hdf5_path = h5py.File(csv_hdf5_mapping_dict['hdf5_path'], 'r')
        """
        self.hdf5_path = csv_hdf5_mapping_dict['hdf5_path']

        self.combo_num = csv_hdf5_mapping_dict['combo_num']
        self.length = csv_hdf5_mapping_dict['length']
        self.n_channels = csv_hdf5_mapping_dict['n_channels']
        self.n_classes = csv_hdf5_mapping_dict['n_classes']
        self.duplicate_num = csv_hdf5_mapping_dict['duplicate_num']
        
        self.complex_aug_prob = csv_hdf5_mapping_dict['complex_aug_prob']
        self.shift_aug_prob = csv_hdf5_mapping_dict['shift_aug_prob']

        self.indexes = np.arange(len(self.csv_file))
        
        shuffle(self.indexes)

    def __call__(self):
        # shuffle
        while True:
            shuffle(self.indexes)
            for idx in range(0,len(self.indexes) - self.combo_num):
                aug_choice = np.random.choice( ['simple_shift','complex_aug'], p=[self.shift_aug_prob,self.complex_aug_prob])
                
                if aug_choice == 'simple_shift':
                    choice_id = self.indexes[idx]
                    choice_line = self.csv_file.iloc[choice_id]
                    key = choice_line[self.key_str]
                    p_t = int(choice_line[self.p_str])
                    s_t = int(choice_line[self.s_str])
                    part = None
                    if self.has_parts:
                        part = choice_line[self.part_str]

                    if self.name == 'DiTing':
                        key_correct = key.split('.')
                        key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
                        p_t = int(p_t)*2
                        s_t = int(s_t)*2
                    
                    X, Y = get_shifted_instance_for_EqDetPhasePicking_training(dataset_name=self.name, 
                                                                                dataset_path=self.hdf5_path,
                                                                                data_length=self.length,
                                                                                key = key,
                                                                                part = part,
                                                                                P = p_t,
                                                                                S = s_t
                                                                                )
                elif aug_choice == 'complex_aug':
                    choice_ids = self.indexes[idx*self.combo_num:(idx+ 1)*self.combo_num]
                    temp_part_list = list()
                    key_list = list()
                    P_list = list()
                    S_list = list()

                    for choice_id in choice_ids:
                        choice_line = self.csv_file.iloc[choice_id]
                        if self.has_parts:
                            part = choice_line[self.part_str]
                            temp_part_list.append(part)
                        
                        key = choice_line[self.key_str]
                        p_t = int(choice_line[self.p_str])
                        s_t = int(choice_line[self.s_str])

                        if self.name == 'DiTing':
                            key_correct = key.split('.')
                            key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
                            p_t = int(p_t)*2
                            s_t = int(s_t)*2
                            
                        key_list.append(key)
                        P_list.append(p_t)
                        S_list.append(s_t)

                        X, Y = get_augmented_instance_for_EqDetPhasePicking_training(dataset_name=self.name,
                                                                                dataset_path=self.hdf5_path,
                                                                                data_length = self.length,
                                                                                temp_part_list = temp_part_list,
                                                                                key_list = key_list,
                                                                                P_list = P_list,
                                                                                S_list = S_list)

                # filter
                X = DiTing_random_filter_augmentation(X)
                
                # channel dropout
                X = DiTing_random_channel_dropout_augmentation(X)

                # duplicate for side-outputs
                Y = np.repeat(Y, self.duplicate_num, axis=-1)
                Y_dict = dict()
                for class_dx in range(self.n_classes):
                    for dup_dx in range(self.duplicate_num):
                        Y_dict['C{}D{}'.format(class_dx,dup_dx)] = Y[:,class_dx*self.duplicate_num+dup_dx:class_dx*self.duplicate_num+dup_dx+1]
                
                yield X, Y_dict

    def on_epoch_end(self):
        shuffle(self.indexes)

class DiTingRealNoiseGenerator:
    # init function
    def __init__(self, csv_hdf5_mapping_dict):
        self.csv_file = pd.read_csv(csv_hdf5_mapping_dict['csv_path'], dtype = {'key': str})
        self.name = csv_hdf5_mapping_dict['name']
        if self.name not in ['TLSC_NO', 'STEAD_NO', 'INSTANCE_NO']:
            print('Dataset type not Supported Yet!!!')
        self.hdf5_path = csv_hdf5_mapping_dict['hdf5_path']
        self.combo_num = csv_hdf5_mapping_dict['combo_num']
        self.length = csv_hdf5_mapping_dict['length']
        self.n_classes = csv_hdf5_mapping_dict['n_classes']
        self.duplicate_num = csv_hdf5_mapping_dict['duplicate_num']
        self.indexes = np.arange(len(self.csv_file))
        self.on_epoch_end()
    def __call__(self):
        # shuffle
        while True:
            shuffle(self.indexes)
            for idx in range(0,len(self.indexes),self.combo_num):
                choice_ids = self.indexes[idx*self.combo_num:(idx+ 1)*self.combo_num]
                if self.name == 'TLSC_NO':
                    pass # private dataset choice_keys = self.csv_file['key'].iloc[choice_ids]
                elif self.name == 'STEAD_NO' or self.name == 'INSTANCE_NO':
                    choice_keys = self.csv_file['trace_name'].iloc[choice_ids]
                X, Y = get_real_noise_for_EqDetPhasePicking_training(dataset_name=self.name,
                                                                     dataset_path = self.hdf5_path,
                                                                     data_length = self.length,
                                                                     key_list = choice_keys)
                X = DiTing_random_filter_augmentation(X)
                X = DiTing_random_channel_dropout_augmentation(X)
                Y = np.repeat(Y, self.duplicate_num, axis=-1)
                Y_dict = dict()
                for class_dx in range(self.n_classes):
                    for dup_dx in range(self.duplicate_num):
                        Y_dict['C{}D{}'.format(class_dx,dup_dx)] = Y[:,class_dx*self.duplicate_num+dup_dx:class_dx*self.duplicate_num+dup_dx+1]
                yield X, Y_dict
    def on_epoch_end(self):
        shuffle(self.indexes)

class DiTingSynNoiseGenerator:
    # init function
    def __init__(self, csv_hdf5_mapping_dict):
        # workaround
        self.noise_length = csv_hdf5_mapping_dict['length']
        self.duplicate_num = csv_hdf5_mapping_dict['duplicate_num']
        self.n_classes = csv_hdf5_mapping_dict['n_classes']
    def __call__(self):
        while True:
            X, Y = get_DiTing_EDPP_Negtive_example(self.noise_length)
            Y = np.repeat(Y, self.duplicate_num, axis=-1)
            Y_dict = dict()
            for class_dx in range(self.n_classes):
                for dup_dx in range(self.duplicate_num):
                    Y_dict['C{}D{}'.format(class_dx,dup_dx)] = Y[:,class_dx*self.duplicate_num+dup_dx:class_dx*self.duplicate_num+dup_dx+1]
            yield X, Y_dict
    def on_epoch_end(self):
        return

####################################################
# Functions for creating final training dataset
####################################################

def get_det_pick_training_dataset(cfgs):
    """
    input: yaml configurations
    output: tf.dataset.Dataset
    """
    duplicate_num = cfgs['Training']['Model']['duplicate_num']
    model_input_length = cfgs['Training']['Model']['input_length']
    model_input_channel = cfgs['Training']['Model']['input_channel']
    
    training_dict_list = []
    training_weight_list = []

    validation_dict_list = []
    validation_weight_list = []
    
    label_type_dict = dict()
    label_shape_dict = dict()
    
    for class_dx in range(cfgs['Training']['Model']['n_classes']):
        for dup_dx in range(duplicate_num):
            label_type_dict['C{}D{}'.format(class_dx,dup_dx)] = tf.float32
            label_shape_dict['C{}D{}'.format(class_dx,dup_dx)] = (cfgs['Training']['Model']['input_length'], 1)

    for dataset_key in cfgs['Training']['Datasets']:
        t_dict = cfgs['Training']['Datasets'][dataset_key]
        t_dict['csv_path'] = t_dict['train_csv_path']
        t_dict['hdf5_path'] = t_dict['train_hdf5_path']
        t_dict['duplicate_num'] = duplicate_num

        if t_dict['has_parts'] is True:
            t_dict['part_list'] = [t_dict['hdf5_path'].format(i) for i in range(t_dict['part_num'])]
        
        training_dict_list.append(t_dict)
        training_weight_list.append(float(t_dict['sample_weight']))

        if t_dict['has_validation'] is True:
            t_dict['csv_path'] = t_dict['val_csv_path']
            t_dict['hdf5_path'] = t_dict['val_hdf5_path']
            if t_dict['has_parts'] is True:
                t_dict['part_list'] = [t_dict['hdf5_path'].format(i) for i in range(t_dict['part_num'])]
                
        validation_dict_list.append(t_dict)
        validation_weight_list.append(float(t_dict['sample_weight']))

    training_dataset_list = [tf.data.Dataset.from_generator(DiTingGenerator(training_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (model_input_length,model_input_channel), label_shape_dict)) for idx in range(len(training_dict_list))]
    training_sample_dataset = tf.data.experimental.sample_from_datasets(training_dataset_list, weights=training_weight_list)

    if len(validation_dict_list) >= 1:
        validation_dataset_list = [tf.data.Dataset.from_generator(DiTingGenerator(validation_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (validation_dict_list[idx]['length'],validation_dict_list[idx]['n_channels']), label_shape_dict)) for idx in range(len(validation_dict_list))]
        validation_sample_dataset = tf.data.experimental.sample_from_datasets(validation_dataset_list, weights=validation_weight_list)
    else:
        validation_sample_dataset = None


    noiseset_dict_list = list()
    noiseset_weight_list = list()
    for noiseset_key in cfgs['Training']['Noisesets']:
        t_dict = cfgs['Training']['Noisesets'][noiseset_key]
        t_dict['duplicate_num'] = duplicate_num
        t_dict['length'] = model_input_length
        noiseset_dict_list.append(t_dict)
        noiseset_weight_list.append(float(t_dict['sample_weight']))
    
    real_negative_dataset_list = [tf.data.Dataset.from_generator(DiTingRealNoiseGenerator(noiseset_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (model_input_length,model_input_channel), label_shape_dict)) for idx in range(len(noiseset_dict_list))]
    real_negative_dataset = tf.data.experimental.sample_from_datasets(real_negative_dataset_list, weights=noiseset_weight_list)
    
    syn_negative_dataset = tf.data.Dataset.from_generator(DiTingSynNoiseGenerator(t_dict), output_types = (tf.float32, label_type_dict), output_shapes = ((model_input_length,model_input_channel), label_shape_dict))
    
    final_training_sample_dataset = tf.data.experimental.sample_from_datasets([training_sample_dataset,real_negative_dataset,syn_negative_dataset], weights=[cfgs['Training']['trace_weight'], cfgs['Training']['real_noise_weight'],cfgs['Training']['syn_noise_weight']])

    final_validation_sample_dataset = tf.data.experimental.sample_from_datasets([validation_sample_dataset, real_negative_dataset.repeat(),syn_negative_dataset.repeat()], weights=[cfgs['Training']['trace_weight'], cfgs['Training']['real_noise_weight'],cfgs['Training']['syn_noise_weight']])

    return final_training_sample_dataset, final_validation_sample_dataset

####################################################
# misc functions
####################################################

def get_triangle_label(label_array, phase_pose, label_length = 101):
    tri_time = np.arange(-0.5,0.5,1/label_length)
    tri_value = np.abs(signal.sawtooth(2 * np.pi* tri_time))
    try:
        label_array[phase_pose - label_length//2: phase_pose - label_length//2 + label_length] = tri_value
    except:
        try:
            label_array[phase_pose] = 1.0
        except:
            pass
    return label_array

def label_EqDetPick(p_t, s_t, data_length, shift = 0, label_length = 101):

    label_P = np.zeros(data_length)
    label_P = get_triangle_label(label_P, p_t + shift, label_length)

    label_S = np.zeros(data_length)
    label_S = get_triangle_label(label_S, s_t + shift, label_length)
    
    label_D = np.zeros(data_length)
    label_D[p_t + shift: s_t + shift + 1] = 1

    return label_P, label_S, label_D

def DiTing_random_channel_dropout_augmentation(X):
    drop_choice = np.random.choice([0, 1, 2, None], p=[0.15,0.15,0.15,0.55])
    if drop_choice == 0:
        X[:,drop_choice] = 0.0
    elif drop_choice == 1:
        X[:,drop_choice] = 0.0
    elif drop_choice == 2:
        X[:,drop_choice] = 0.0
    else:
        pass
    return X

def DiTing_random_filter_augmentation(X):
    filter_choice = np.random.choice(['fix', 'high_random', 'low_random', 'none'], p=[0.35,0.15,0.15,0.35])
    if filter_choice == 'none':
        return X
    elif filter_choice == 'fix':
        filtered_X = np.zeros_like(X)
        for chdx in range(3):
            temp_trace = obspy.Trace(data=X[:,chdx])
            temp_trace.stats.sampling_rate = 100.0
            temp_trace.filter('bandpass',freqmin=0.5,freqmax=40,zerophase=True)
            filtered_X[:,chdx] = temp_trace.data[:]
            filtered_X[:,chdx] -= np.mean(filtered_X[:,chdx])
            norm_factor = np.std(filtered_X[:,chdx])
            if norm_factor == 0:
                pass
            else:
                filtered_X[:,chdx] /= norm_factor
    elif filter_choice == 'high_random':
        filtered_X = np.zeros_like(X)
        for chdx in range(3):
            temp_trace = obspy.Trace(data=X[:,chdx])
            temp_trace.stats.sampling_rate = 100.0
            temp_trace.filter('bandpass',freqmin=np.random.uniform(low=0.1,high=2.0),freqmax=np.random.uniform(low=8.0,high=45.0),zerophase=True)
            filtered_X[:,chdx] = temp_trace.data[:]
            filtered_X[:,chdx] -= np.mean(filtered_X[:,chdx])
            norm_factor = np.std(filtered_X[:,chdx])
            if norm_factor == 0:
                pass
            else:
                filtered_X[:,chdx] /= norm_factor
    elif filter_choice == 'low_random':
        filtered_X = np.zeros_like(X)
        for chdx in range(3):
            temp_trace = obspy.Trace(data=X[:,chdx])
            temp_trace.stats.sampling_rate = 100.0
            temp_trace.filter('bandpass',freqmin=np.random.uniform(low=0.001,high=0.5),freqmax=np.random.uniform(low=4.0,high=10.0),zerophase=True)
            filtered_X[:,chdx] = temp_trace.data[:]
            filtered_X[:,chdx] -= np.mean(filtered_X[:,chdx])
            norm_factor = np.std(filtered_X[:,chdx])
            if norm_factor == 0:
                pass
            else:
                filtered_X[:,chdx] /= norm_factor
    return filtered_X