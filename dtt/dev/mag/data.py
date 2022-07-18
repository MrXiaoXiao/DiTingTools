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

def get_from_INSTANCE(key=None, h5file_path=''):
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

####################################################
# Functions for creating a training instance
####################################################
def get_instance_for_MagReg_training(dataset_name = 'DiTing',
                                        dataset_path = None,
                                        part = None,
                                        key = None,
                                        P = None,
                                        length_before_P = 100,
                                        length_after_P = 924,
                                        mag = None
                                        ):
    data_length = int(length_before_P + length_after_P)
    temp_data_X = np.zeros([data_length, 3])
    temp_data_Y = np.zeros([1])

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

    temp_data_X[:,:] = data[P - length_before_P: P + length_after_P, :]
    for chdx in range(3):
        temp_data_X[:,chdx] -= np.mean(temp_data_X[:,chdx])

    temp_data_Y[0] = mag/10.0

    return temp_data_X, temp_data_Y

####################################################
# Functions for creating a training generator
####################################################
class DiTingMagGenerator:
    # init function
    def __init__(self, csv_hdf5_mapping_dict):
        self.csv_file = pd.read_csv(csv_hdf5_mapping_dict['csv_path'], dtype = {'key': str})
        self.name = csv_hdf5_mapping_dict['name']
        if self.name not in ['DiTing', 'INSTANCE', 'STEAD']:
            print('Dataset type not Supported Yet!!!')
        self.has_parts = csv_hdf5_mapping_dict['has_parts']
        self.hdf5_path = csv_hdf5_mapping_dict['hdf5_path']
        
        self.length_before_P = csv_hdf5_mapping_dict['length_before_P']
        self.length_after_P = csv_hdf5_mapping_dict['length_after_P']
        self.n_channels = csv_hdf5_mapping_dict['n_channels']

        # clean wrong baz data
        if self.name == 'DiTing':
            self.csv_file =  self.csv_file[pd.to_numeric(self.csv_file['evmag'], errors='coerce').notnull()]
        elif self.name == 'STEAD':
            self.csv_file =  self.csv_file[pd.to_numeric(self.csv_file['source_magnitude'], errors='coerce').notnull()]
        elif self.name == 'INSTANCE':
            self.csv_file =  self.csv_file[pd.to_numeric(self.csv_file['source_magnitude'], errors='coerce').notnull()]
        else:
            pass

    def __call__(self):
        # shuffle
        indexes = np.arange(len(self.csv_file))
        while True:
            shuffle(indexes)
            
            for idx in range(0,len(indexes)):
                if self.name == 'DiTing':
                    choice_id = indexes[idx]
                    choice_line = self.csv_file.iloc[choice_id]
                    part = choice_line['part']
                    key = choice_line['key']
                    key_correct = key.split('.')
                    key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
                    p_t = int(choice_line['p_pick'] *2)
                    # label to be corrected!
                    mag = float(choice_line['evmag'])
                    
                    X, Y = get_instance_for_MagReg_training(dataset_name=self.name,
                                                            dataset_path = self.hdf5_path,
                                                            length_before_P = self.length_before_P,
                                                            length_after_P = self.length_after_P,
                                                            part = part,
                                                            key = key,
                                                            mag = mag,
                                                            P = p_t)
                    yield X, Y
                
                elif self.name == 'STEAD':
                    choice_id = indexes[idx]
                    key = self.csv_file['trace_name'].iloc[choice_id]
                    p_t = int(self.csv_file['p_arrival_sample'].iloc[choice_id])
                    mag = float(self.csv_file['source_magnitude'].iloc[choice_id])
                    X, Y = get_instance_for_MagReg_training(dataset_name=self.name,
                                                            dataset_path = self.hdf5_path,
                                                            length_before_P = self.length_before_P,
                                                            length_after_P = self.length_after_P,
                                                            key = key,
                                                            mag = mag,
                                                            P = p_t) 
                    yield X, Y
                    
                elif self.name == 'INSTANCE':
                    choice_id = indexes[idx]
                    choice_line = self.csv_file.iloc[choice_id]
                    key = choice_line['trace_name']
                    p_t = int(choice_line['trace_P_arrival_sample'])
                    mag = choice_line['source_magnitude']
                    X, Y = get_instance_for_MagReg_training(dataset_name=self.name,
                                                            dataset_path = self.hdf5_path,
                                                            length_before_P = self.length_before_P,
                                                            length_after_P = self.length_after_P,
                                                            key = key,
                                                            mag = mag,
                                                            P = p_t)
                    yield X, Y
                else:
                    print('Dataset Not Supported!')

####################################################
# Functions for creating final training dataset
####################################################

def get_Mag_training_dataset(cfgs):
    """
    input: yaml configurations
    output: tf.dataset.Dataset
    """
    training_dict_list = []
    training_weight_list = []

    validation_dict_list = []
    validation_weight_list = []
    
    for dataset_key in cfgs['Training']['Datasets']:
        t_dict = cfgs['Training']['Datasets'][dataset_key]
        t_dict['csv_path'] = t_dict['train_csv_path']
        t_dict['hdf5_path'] = t_dict['train_hdf5_path']

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

    training_dataset_list = [tf.data.Dataset.from_generator(DiTingMagGenerator(training_dict_list[idx]),output_types = (tf.float32, tf.float32), output_shapes = ( (training_dict_list[idx]['length_before_P'] + training_dict_list[idx]['length_after_P'],training_dict_list[idx]['n_channels']), 1)) for idx in range(len(training_dict_list))]
    training_sample_dataset = tf.data.experimental.sample_from_datasets(training_dataset_list, weights=training_weight_list)

    if len(validation_dict_list) >= 1:
        validation_dataset_list = [tf.data.Dataset.from_generator(DiTingMagGenerator(validation_dict_list[idx]),output_types = (tf.float32, tf.float32), output_shapes = ( (validation_dict_list[idx]['length_before_P'] + validation_dict_list[idx]['length_after_P'],validation_dict_list[idx]['n_channels']), 1)).repeat() for idx in range(len(validation_dict_list))]
        validation_sample_dataset = tf.data.experimental.sample_from_datasets(validation_dataset_list, weights=validation_weight_list)
    else:
        validation_sample_dataset = None

    return training_sample_dataset, validation_sample_dataset

####################################################
# misc functions
####################################################
