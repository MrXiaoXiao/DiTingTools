from dtt.dev.det_pick.tfDataset import get_EqDetPhasePicking_training_dataset_with_Negtive_sampling
from dtt.dev.det_pick.models.DiTingPicker import DiTingPicker
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import os
import yaml
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

def train(cfgs=None):
    """
    Function for training earthquake detection and phase picking models
    input: cfgs. yaml configuration files
    """
    if cfgs==None:
        print('Empty Config Error')
        return
    if cfgs['Training']['Model']['type'] == 'DiTingPicker':
        model = DiTingPicker(cfgs)
    else:
        print('Model Type Error!')
        return
    print('Done Creating Model')
    if cfgs['Training']['use_pre_trained_model']:
        model.load_weights(cfgs['Training']['PreTrainedPath'])
        print('Done Load weights')
    
    training_dataset, validation_data_gen = get_EqDetPhasePicking_training_dataset_with_Negtive_sampling(cfgs)
    
    print('Done Creating Dataset')

    TASK_ID = cfgs['Training']['TASK_ID']
    if os.path.exists(cfgs['Training']['filepath']):
        pass
    else:
        os.makedirs(cfgs['Training']['filepath'])
    filepath = cfgs['Training']['filepath'] + TASK_ID + '.weights.{epoch:02d}-{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, mode='auto', period=1)
    print('Done Creating Generator')
    
    earlyStoping = EarlyStopping(monitor='val_loss', patience=40)
    lr_reducer = ReduceLROnPlateau(factor=0.5,cooldown=0,patience=5,min_lr = cfgs['Training']['min_lr'])

    hist = model.fit(training_dataset.batch(cfgs['Training']['batchsize']).prefetch(tf.data.experimental.AUTOTUNE),
                    workers=cfgs['Training']['Workers'],
                    use_multiprocessing=cfgs['Training']['multiprocessing'],
                    callbacks=[checkpoint,earlyStoping,lr_reducer],
                    epochs=cfgs['Training']['epochs'],
                    steps_per_epoch=cfgs['Training']['steps_per_epoch'],
                    validation_data=validation_data_gen.batch(cfgs['Training']['batchsize']).prefetch(tf.data.experimental.AUTOTUNE),
                    validation_steps=cfgs['Training']['validation_steps']
                    )

    histpath = cfgs['Training']['histpath']  + TASK_ID + '_hist.pickle'
    with open(histpath, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    model.save(cfgs['Training']['histpath'] + TASK_ID + '_last.hdf5')
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiTingTools: EarthquakeDetectionAndPhasePicking Training')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Path to Configuration file')
    args = parser.parse_args()
    cfgs = yaml.load(open(args.config_file), Loader=yaml.SafeLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs['Training']['GPU_ID']
    train(cfgs)
