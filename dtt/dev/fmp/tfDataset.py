import h5py
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from random import shuffle
from scipy import signal

def get_FMP_training_dataset(cfgs):
    """
    input: yaml configurations
    output: tf.dataset.Dataset. train and validation datasets. 
    """
    duplicate_num = cfgs['Training']['Model']['duplicate_num']
    
    training_dict_list = []
    training_weight_list = []

    validation_dict_list = []
    validation_weight_list = []
    
    label_type_dict = dict()
    label_shape_dict = dict()
    for class_dx in range(2):
        for dup_dx in range(duplicate_num):
            label_type_dict['T{}D{}'.format(class_dx,dup_dx)] = tf.float32
            label_shape_dict['T{}D{}'.format(class_dx,dup_dx)] = 3

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

    training_dataset_list = [tf.data.Dataset.from_generator(DiTingFMPGenerator(training_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (training_dict_list[idx]['length'],training_dict_list[idx]['n_channels']), label_shape_dict)).repeat() for idx in range(len(training_dict_list))]
    training_sample_dataset = tf.data.experimental.sample_from_datasets(training_dataset_list, weights=training_weight_list)

    if len(validation_dict_list) >= 1:
        validation_dataset_list = [tf.data.Dataset.from_generator(DiTingFMPGenerator(validation_dict_list[idx]),output_types = (tf.float32, label_type_dict), output_shapes = ( (validation_dict_list[idx]['length'],validation_dict_list[idx]['n_channels']), label_shape_dict)) for idx in range(len(validation_dict_list))]
        validation_sample_dataset = tf.data.experimental.sample_from_datasets(validation_dataset_list, weights=validation_weight_list)
    else:
        validation_sample_dataset = None

    return training_sample_dataset, validation_sample_dataset

def get_instance_for_FirstMotionPolarity_training(dataset_name='DiTing',
                                                dataset_path = None,
                                                data_length = 128,
                                                part = None,
                                                key = None,
                                                motion = None,
                                                sharpness = None,
                                                P = None):
    """
    Get training instance for First Motion Polarity training
    """
    half_len = data_length//2
    temp_data_X = np.zeros([int(data_length),2])
    temp_data_Y = np.zeros([2,3])

    if dataset_name == 'DiTing':
        try:
            dataset = part.get('earthquake/'+str(key))    
            data = np.array(dataset).astype(np.float32)
            up_sample_data = np.zeros([len(data)*2, 3])
            # upsampling 50 Hz to 100 Hz
            for chdx in range(3):
                up_sample_data[:,chdx] = signal.resample(data[:,chdx], len(data[:,chdx]) * 2)
            data = up_sample_data
        except:
            print('Error on key: {}'.format(key))
            return temp_data_X, temp_data_Y
        p_t = int(P)
        if p_t < 0 or p_t > 18000:
            return temp_data_X, temp_data_Y
        temp_data_X[:,0] = data[p_t - half_len: p_t + half_len, 0]
    else:
        print('Dataset Type Not Supported!!!')
        return
    
    temp_data_X[:,0] -= np.mean(temp_data_X[:,0])
    norm_factor = np.max(np.abs(temp_data_X[:,0]))
    if norm_factor == 0:
        pass
    else:
        temp_data_X[:,0] /= norm_factor
    
    reverse_factor = np.random.choice([-1,1])
    rescale_factor = np.random.uniform(low=0.5,high=1.5)
    
    temp_data_X[:,0] *= reverse_factor
    temp_data_X[:,0] *= rescale_factor
    
    diff_data = np.diff(temp_data_X[half_len:,0])
    diff_sign_data = np.sign(diff_data)
    temp_data_X[half_len+1:,1] = diff_sign_data[:]
    
    if motion == 'U' or motion == 'C':
        if reverse_factor == 1:
            temp_data_Y[0,0] = 1
        else:
            temp_data_Y[0,1] = 1
    elif motion == 'R' or motion == 'D':
        if reverse_factor == 1:
            temp_data_Y[0,1] = 1
        else:
            temp_data_Y[0,0] = 1
            
    if sharpness == 'I':
        temp_data_Y[1,0] = 1
    else:
        temp_data_Y[1,1] = 1

    return temp_data_X, temp_data_Y
    
class DiTingFMPGenerator:
    """
    Generator for first motion polarity determination
    """
    def __init__(self, csv_hdf5_mapping_dict):
        csv_file = pd.read_csv(csv_hdf5_mapping_dict['csv_path'], dtype = {'key': str})
        self.csv_file  = csv_file[csv_file['p_motion'].str.contains('C') | csv_file['p_motion'].str.contains('R') | csv_file['p_motion'].str.contains('U') | csv_file['p_motion'].str.contains('D')]
        self.name = csv_hdf5_mapping_dict['name']
        if self.name not in ['DiTing']:
            print('Dataset type not Supported Yet!!!')
        self.has_parts = csv_hdf5_mapping_dict['has_parts']
        if self.has_parts:
            self.part_num = csv_hdf5_mapping_dict['part_num']
            self.part_list = []
            for idx in range(self.part_num):
                self.part_list.append( h5py.File(csv_hdf5_mapping_dict['part_list'][idx], 'r') )
        else:
            self.hdf5_path = h5py.File(csv_hdf5_mapping_dict['hdf5_path'], 'r')
        
        self.length = csv_hdf5_mapping_dict['length']
        self.n_channels = 2
        self.n_classes = 3
        self.n_predvar = 2
        self.duplicate_num = csv_hdf5_mapping_dict['duplicate_num']

    def __call__(self):
        # shuffle
        indexes = np.arange(len(self.csv_file))
        #while True:
        shuffle(indexes)
        
        for idx in range(0,len(indexes)):
            if self.name == 'DiTing':
                choice_id = indexes[idx]
                choice_line = self.csv_file.iloc[choice_id]
                part = self.part_list[choice_line['part']]
                key = choice_line['key']
                key_correct = key.split('.')
                key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
                # 2 x phase label upsample
                p_t = choice_line['p_pick']*2
                motin = choice_line['p_motion']
                sharpness = choice_line['p_clarity']
                X, Y = get_instance_for_FirstMotionPolarity_training(dataset_name=self.name,
                                                data_length = self.length,
                                                part = part,
                                                key = key,
                                                motion = motin,
                                                sharpness = sharpness,
                                                P = p_t)

                Y_dict = dict()
                for type_dx in range(self.n_predvar):
                    for dup_dx in range(self.duplicate_num):
                        Y_dict['T{}D{}'.format(type_dx,dup_dx)] = Y[type_dx,:]
                yield X, Y_dict
            else:
                pass