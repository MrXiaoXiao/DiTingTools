"""
Return the Dataset for training
"""
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

def get_EqDetPhasePicking_training_dataset(cfgs):
    """
    Get dataset for training Earthquake detection and phase picking
    input: cfgs (yaml configurations) Please see the example yaml file for details
    output: tf.dataset.Dataset
    """
    duplicate_num = cfgs['Training']['Model']['duplicate_num']
    
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

    training_dataset_list = [tf.data.Dataset.from_generator(DiTingGenerator(training_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (training_dict_list[idx]['length'],training_dict_list[idx]['n_channels']), label_shape_dict)) for idx in range(len(training_dict_list))]
    training_sample_dataset = tf.data.experimental.sample_from_datasets(training_dataset_list, weights=training_weight_list)
    
    if len(validation_dict_list) >= 1:
        validation_dataset_list = [tf.data.Dataset.from_generator(DiTingGenerator(validation_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (validation_dict_list[idx]['length'],validation_dict_list[idx]['n_channels']), label_shape_dict)) for idx in range(len(validation_dict_list))]
        validation_sample_dataset = tf.data.experimental.sample_from_datasets(validation_dataset_list, weights=validation_weight_list)
    else:
        validation_sample_dataset = None

    return training_sample_dataset, validation_sample_dataset

class DiTingGenerator:
    """
    Generator function for earthquake detection and phase picking using DiTing Dataset
    """
    def __init__(self, csv_hdf5_mapping_dict):
        """
        init 
        """
        self.csv_file = pd.read_csv(csv_hdf5_mapping_dict['csv_path'], dtype = {'key': str})
        self.name = csv_hdf5_mapping_dict['name']
        if self.name not in ['DiTing', 'STEAD', 'INSTANCE']:
            print('Dataset type not Supported Yet!!!')
        self.has_parts = csv_hdf5_mapping_dict['has_parts']
        if self.has_parts:
            self.part_num = csv_hdf5_mapping_dict['part_num']
            self.part_list = []
            for idx in range(self.part_num):
                self.part_list.append( h5py.File(csv_hdf5_mapping_dict['part_list'][idx], 'r') )
        else:
            self.hdf5_path = h5py.File(csv_hdf5_mapping_dict['hdf5_path'], 'r')
        self.combo_num = csv_hdf5_mapping_dict['combo_num']
        self.length = csv_hdf5_mapping_dict['length']
        self.n_channels = csv_hdf5_mapping_dict['n_channels']
        self.n_classes = csv_hdf5_mapping_dict['n_classes']
        self.duplicate_num = csv_hdf5_mapping_dict['duplicate_num']
        self.indexes = np.arange(len(self.csv_file))
    def __call__(self):
        while True:
            shuffle(self.indexes)
            for idx in range(0,len(self.indexes),self.combo_num):
                if self.name == 'DiTing':
                    choice_ids = self.indexes[idx*self.combo_num:(idx+ 1)*self.combo_num]
                    temp_part_list = list()
                    key_list = list()
                    P_list = list()
                    S_list = list()

                    for choice_id in choice_ids:
                        choice_line = self.csv_file.iloc[choice_id]
                        part = choice_line['part']
                        key = choice_line['key']
                        key_correct = key.split('.')
                        key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')

                        p_t = choice_line['p_pick']
                        s_t = choice_line['s_pick']

                        temp_part_list.append(self.part_list[part])
                        key_list.append(key)
                        P_list.append(p_t)
                        S_list.append(s_t)
                    
                    X, Y = get_instance_for_EqDetPhasePicking_training(dataset_name=self.name,
                                                data_length = self.length,
                                                temp_part_list = temp_part_list,
                                                key_list = key_list,
                                                P_list = P_list,
                                                S_list = S_list)
                    X = DiTing_random_filter_augmentation(X)
                    X = DiTing_random_channel_dropout_augmentation(X)

                    Y = np.repeat(Y, self.duplicate_num, axis=-1)
                    Y_dict = dict()
                    for class_dx in range(self.n_classes):
                        for dup_dx in range(self.duplicate_num):
                            Y_dict['C{}D{}'.format(class_dx,dup_dx)] = Y[:,class_dx*self.duplicate_num+dup_dx:class_dx*self.duplicate_num+dup_dx+1]
                    
                    yield X, Y_dict
                    
                elif self.name == 'STEAD':
                    choice_ids = self.indexes[idx*self.combo_num:(idx+ 1)*self.combo_num]
                    choice_keys = self.csv_file['trace_name'].iloc[choice_ids]
                    X, Y = get_instance_for_EqDetPhasePicking_training(dataset_name = self.name,
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
                    
                elif self.name == 'INSTANCE':
                    choice_ids = self.indexes[idx*self.combo_num:(idx+ 1)*self.combo_num]
                    key_list = []
                    P_list = []
                    S_list = []
                    for choice_id in choice_ids:
                        choice_line = self.csv_file.iloc[choice_id]
                        key = choice_line['trace_name']
                        p_t = choice_line['trace_P_arrival_sample']
                        s_t = choice_line['trace_S_arrival_sample']

                        if (s_t - p_t) > 10:
                            pass
                        else:
                            continue

                        key_list.append(key)
                        P_list.append(p_t)
                        S_list.append(s_t)
                
                    X, Y = get_instance_for_EqDetPhasePicking_training(dataset_name=self.name,
                                                                        dataset_path = self.hdf5_path,
                                                                        data_length = self.length,
                                                                        key_list = key_list,
                                                                        P_list = P_list,
                                                                        S_list = S_list)
                    X = DiTing_random_filter_augmentation(X)
                    X = DiTing_random_channel_dropout_augmentation(X)

                    Y = np.repeat(Y, self.duplicate_num, axis=-1)
                    Y_dict = dict()
                    for class_dx in range(self.n_classes):
                        for dup_dx in range(self.duplicate_num):
                            Y_dict['C{}D{}'.format(class_dx,dup_dx)] = Y[:,class_dx*self.duplicate_num+dup_dx:class_dx*self.duplicate_num+dup_dx+1]
                    yield X, Y_dict
                else:
                    print('Dataset error!!!')
                    pass

def gen_colored_noise(alpha, length, dt = 0.01, fmin=0):
    """
    input:
        alpha: float. alpha value of the colored noise
        length: int. the lenght of noise data
        dt: float. sampling delta t.
        fmin: float. the minimum frequency in generated noise.
    output:
        noise data array of the input length. 
    """

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

def simple_label(label_array, phase_pose):
    """
    Create a simple label for phase picking
    input:
        label_array: zero array.
        phase_pose: the label position.
    output:
        label_array. 
    """
    if phase_pose + 10 <= len(label_array):
        label_array[phase_pose] = 1.0
        label_array[phase_pose+1] = 0.8
        label_array[phase_pose-1] = 0.8
        label_array[phase_pose+2] = 0.6
        label_array[phase_pose-2] = 0.6
        label_array[phase_pose+3] = 0.4
        label_array[phase_pose-3] = 0.4
        label_array[phase_pose+4] = 0.2
        label_array[phase_pose-4] = 0.2
    return label_array



def DiTing_random_channel_dropout_augmentation(X):
    """
    channel dropout augmentation
    input:
        X: training input instance array
    output:
        X after channel dropout augmentation
    """
    drop_choice = np.random.choice([0, 1, 2, None, None, None])
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
    """
    random filter augmentation
    input:
        X: training input instance array
    output:
        X after random filter augmentation
    """
    filter_choice = np.random.choice(['fix', 'high_random', 'low_random', 'none'])

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

def get_instance_for_EqDetPhasePicking_training(dataset_name='DiTing',
                                                dataset_path = None,
                                                data_length = 6144,
                                                temp_part_list = None,
                                                key_list = None,
                                                P_list = None,
                                                S_list = None,
                                                pad_noise_prob = 0.2,
                                                min_noise_len = 200,
                                                max_noise_len = 1000,
                                                max_time = 2
                                                ):
    """
    Get training instances for earthquake detection and phase picking
    input:
        dataset_name: the name of the dataset type
        dataset_path: the full path to the dataset
        data_length: the length of the input data
        temp_part_list: Optional. The part list of the DiTing Dataset.
        key_list: Optional.
        P_list:
        S_list:
        pad_noise_prob:
        min_noise_len:
        max_noise_len:
        max_time:
    output:
        temp_data_X: train input instance
        temp_data_Y: train label instance
    """
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
                    
        if dataset_name == 'DiTing':
            try:
                dataset = temp_part_list[key_dx].get('earthquake/'+str(key))    
                data = np.array(dataset).astype(np.float32)
                up_sample_data = np.zeros([len(data)*2, 3])
                # upsampling 50 Hz to 100 Hz
                for chdx in range(3):
                    up_sample_data[:,chdx] = signal.resample(data[:,chdx], len(data[:,chdx]) * 2)
                data = up_sample_data
            except:
                print('Error on key {} DiTing'.format(key))
                continue
            # upsample label
            p_t = int(P_list[key_dx])*2
            s_t = int(S_list[key_dx])*2
            
            # rule out error data
            if p_t < 0 or p_t > 18000 or s_t < 0 or s_t > 18000:
                continue
            
            # shift and crop data
            p_shift = np.random.randint(high=3000,low=500)
            s_shift = np.random.randint(high=int(6*(s_t-p_t)),low=int(4*(s_t-p_t)))
            if p_t - p_shift <= 0:
                p_shift = p_t - 100
            if s_t + s_shift >= len(data):
                s_shift = len(data) - s_t - 200
        
        elif dataset_name == 'STEAD':
            try:
                dataset = dataset_path.get('earthquake/local/'+str(key))
                data = np.array(dataset).astype(np.float32)
            except:
                print('Error on key {} STEAD'.format(key))
                continue
            p_t = int(dataset.attrs['p_arrival_sample'])
            s_t = int(dataset.attrs['s_arrival_sample'])

            p_shift = np.random.randint(high=2000,low=500)
            s_shift = np.random.randint(high=int(14*(s_t-p_t)),low=int(8*(s_t-p_t)))
            if p_t - p_shift <= 0:
                p_shift = p_t - 100
            if s_t + s_shift >= len(data):
                s_shift = len(data) - s_t - 200
        
        elif dataset_name == 'INSTANCE':
            try:
                dataset = dataset_path.get('data/'+str(key))
                data_t = np.array(dataset).astype(np.float32)
            except:
                print('Error on key {} INSTANCE'.format(key))
                continue
            #print(np.shape(data))
            data = np.zeros([12000,3])
            data[:,0] = data_t[2,:]
            data[:,1] = data_t[1,:]
            data[:,2] = data_t[0,:]
            #data = np.reshape(data, [12000,3])
            p_t = int(P_list[key_dx])
            s_t = int(S_list[key_dx])
            p_shift = np.random.randint(high=2000,low=500)
            s_shift = np.random.randint(high=int(10*(s_t-p_t)),low=int(6*(s_t-p_t)))
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

        label_P = np.zeros(len(data))
        label_P = simple_label(label_P, p_shift)

        label_S = np.zeros(len(data))
        label_S = simple_label(label_S, p_shift + s_t - p_t)
        
        label_d = np.zeros(len(data))
        label_d[p_shift:p_shift + s_t - p_t + 1] = 1
        
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
        temp_data_Y[start_index:start_index + pad_len,2] = label_d[:pad_len]
        start_index += pad_len
    # workaround factor
    shift_dx = np.where(temp_data_Y[:data_length*int(max_time-1),2]==0)[0]
    shift_dx = np.random.choice(shift_dx)
    #shift_dx = np.random.randint(low=X,high=X)
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
    """
    Function for generating a varity types of noise
    input:
        noise_length: the name of the dataset type
    output:
        noise_data: noise input instance
        noise_Y: zero array. noise label instance
    """
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

class DiTingNegSampleGenerator:
    """
    Generator class for noise instance
    """
    # init function
    def __init__(self, cfgs=None):
        #TODO add cfgs to neg
        # workaround
        self.noise_length = 6144
        self.duplicate_num = 4
        self.n_classes = 3
    def __call__(self):
        while True:
            X, Y = get_DiTing_EDPP_Negtive_example(self.noise_length)
            Y = np.repeat(Y, self.duplicate_num, axis=-1)
            Y_dict = dict()
            for class_dx in range(self.n_classes):
                for dup_dx in range(self.duplicate_num):
                    Y_dict['C{}D{}'.format(class_dx,dup_dx)] = Y[:,class_dx*self.duplicate_num+dup_dx:class_dx*self.duplicate_num+dup_dx+1]
            yield X, Y_dict

def get_EqDetPhasePicking_training_dataset_with_Negtive_sampling(cfgs):
    """
    input: yaml configurations
    output: tf.dataset.Dataset
    """
    duplicate_num = cfgs['Training']['Model']['duplicate_num']
    
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

    training_dataset_list = [tf.data.Dataset.from_generator(DiTingGenerator(training_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (training_dict_list[idx]['length'],training_dict_list[idx]['n_channels']), label_shape_dict)) for idx in range(len(training_dict_list))]
    training_sample_dataset = tf.data.experimental.sample_from_datasets(training_dataset_list, weights=training_weight_list)
    
    negtive_dataset = tf.data.Dataset.from_generator(DiTingNegSampleGenerator(), output_types = (tf.float32, label_type_dict), output_shapes = ((6144,3), label_shape_dict))
    
    final_training_sample_dataset = tf.data.experimental.sample_from_datasets([training_sample_dataset, negtive_dataset], weights=[0.8, 0.2])
    
    if len(validation_dict_list) >= 1:
        validation_dataset_list = [tf.data.Dataset.from_generator(DiTingGenerator(validation_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (validation_dict_list[idx]['length'],validation_dict_list[idx]['n_channels']), label_shape_dict)) for idx in range(len(validation_dict_list))]
        validation_sample_dataset = tf.data.experimental.sample_from_datasets(validation_dataset_list, weights=validation_weight_list)
    else:
        validation_sample_dataset = None
    
    final_validation_sample_dataset = tf.data.experimental.sample_from_datasets([validation_sample_dataset, negtive_dataset], weights=[0.8, 0.2])

    return final_training_sample_dataset, final_validation_sample_dataset

def get_periodic_energy(noise_length):
    """
    Function for generating periodic energy noise
    input: noise_length
    output: noise array
    """
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
    """
    Function for generating long rectangle noise
    input: noise_length
    output: noise array
    """
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
    """
    Function for generating disalignment misguide noise
    input: noise_length
    output: noise array
    """
    noise_data = np.zeros([noise_length, 3])
    shift_index = np.random.randint(low=0, high=int(noise_length*0.8))
    alpha = np.random.uniform(low=0,high=4)
    taper_value = np.random.uniform(low=0, high=0.1)
    
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
    """
    Function for generating artifical boundary noise
    input: noise_length
    output: noise array
    """
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
    """
    Function for generating ['sine', 'triangle', 'spike', 'rect', 'ricker', 'random', 'empty'] noise
    input: noise_length
    output: noise array
    """
    misguide_P_pos = np.random.randint(low=0,high=noise_length - misguide_width_max*8)
    misguide_S_pos = np.random.randint(low=misguide_P_pos + 10,high=noise_length - misguide_width_max*4)
    misguide_P_decay = np.random.uniform(low=0.1,high=2.0)
    misguide_S_decay = np.random.uniform(low=0.1,high=2.0)
    
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

