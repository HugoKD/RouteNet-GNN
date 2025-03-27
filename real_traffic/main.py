import os
import tensorflow as tf
from data_generator import input_fn
from time import sleep
import sys
import time
from delay_model import RouteNet_Fermi as Routenet_Fermi_GRU
from delay_model_LSTM import RouteNet_Fermi as Routenet_Fermi_LSTM
import json
TRAIN_PATH = '../data/TON23/real_traces/train/geant'
VALIDATION_PATH = '../data/TON23/real_traces/validation'
TEST_PATH = '../data/TON23/real_traces/test/geant'


ds_train = input_fn(TRAIN_PATH, shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)
ds_validation = input_fn(VALIDATION_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = input_fn(TEST_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)

def get_len():

    for type in ['train', 'validation', 'test']:
        counter = 0
        dataset = globals()[f'ds_{type}']
        for _ in dataset:
            counter += 1
        print(f'il y a {counter} éléments dans {type}_ds')


def train(epoch = 20) :
    type_training=['GRU', 'LSTM']
    for type in type_training:
        TRAIN_PATH = '../data/TON23/real_traces/train/geant'
        VALIDATION_PATH = '../data/TON23/real_traces/validation'
        TEST_PATH = '../data/TON23/real_traces/test/geant'

        b = time.time()
        # 📦 Chargement des datasets
        ds_train = input_fn(TRAIN_PATH, shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)
        ds_validation = input_fn(VALIDATION_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = input_fn(TEST_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)


        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        if type == 'GRU' :
            model = Routenet_Fermi_GRU()
        else :
            model = Routenet_Fermi_LSTM()

        loss_object = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(loss=loss_object,
                      optimizer=optimizer,
                      run_eagerly=False)
        if type == 'LSTM' :
            ckpt_dir = './ckpt_dir_der_des_der_LSTM'
        elif type == 'GRU' :
            ckpt_dir = './ckpt_dir_der_des_der_GRU'

        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is not None:
            print("Found a pretrained model, restoring...")
            model.load_weights(latest)
        else:
            print("Starting training from scratch...")

        filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            verbose=10,
            mode="min",
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch')


        history = model.fit(ds_train,
                  epochs=epoch,
                  validation_data=ds_test,
                  callbacks=[cp_callback],
                  use_multiprocessing=True)

        end_training = time.time()
        print(f"Temps écoulé : {end_training - b:.2f} secondes")
        model.evaluate(ds_test)

        if type == 'GRU' :
            history_dict = history.history
            # Save it under the form of a json file
            json.dump(history_dict, open('history_GRU', 'w'))
        elif type == 'LSTM' :
            history_dict = history.history
            # Save it under the form of a json file
            json.dump(history_dict, open('history_LSTM', 'w'))

#train(epoch=20)
get_len()