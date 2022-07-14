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

def get_from_DiTing_FMP(part=None, key=None, h5file_path=''):
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

def get_from_SCSN_FMP(index, h5file_path=''):
    """
    get waveform from SCSN FMP
    TODO: description
    """
    with h5py.File(h5file_path, 'r') as f:
        data = np.asarray(f['X'][index])
        label = np.asarray(f['Y'][index])
    return data, label

####################################################
# Functions for creating a training instance
####################################################
def get_augmented_instance_for_FMP_training(dataset_name='DiTing',
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
class DiTingGenerator_FMP:
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

####################################################
# Functions for creating final training dataset
####################################################

def get_FMP_training_dataset(cfgs):
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
