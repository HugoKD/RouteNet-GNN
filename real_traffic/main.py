import os
import tensorflow as tf
from data_generator import input_fn
from time import sleep
import sys
import time
from delay_model import RouteNet_Fermi as Routenet_Fermi_GRU
from delay_model_LSTM import RouteNet_Fermi as Routenet_Fermi_LSTM
import json
import matplotlib.pyplot as plt

"""
Script to train both GRU and LSTM variants simultaneously using default parameters for RouteNet-Fermi over X epochs.
The dataset must be located in: ~/data/TON23/real_traces/{test, train, validation}
"""



TRAIN_PATH = '../data/TON23/real_traces/train/geant'
VALIDATION_PATH = '../data/TON23/real_traces/validation'
TEST_PATH = '../data/TON23/real_traces/test/geant'


ds_train = input_fn(TRAIN_PATH, shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)
ds_validation = input_fn(VALIDATION_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = input_fn(TEST_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)

def get_len():
    '''
    Obtenir la longueur de chaque dataset (seule solution)
    '''
    for type in ['train', 'validation', 'test']:
        counter = 0
        dataset = globals()[f'ds_{type}']
        for _ in dataset:
            counter += 1
        print(f'il y a {counter} éléments dans {type}_ds')


def train_gru_and_lstm(epoch = 2) :
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


def train_tweak_features(features, epoch=5):
    """
    Entrainer le modèles sur différents features pour ensuite les analyser dans soutenance.py
    """
    # Chemins vers les datasets
    TRAIN_PATH = '../data/TON23/real_traces/train/geant'
    VALIDATION_PATH = '../data/TON23/real_traces/validation'
    TEST_PATH = '../data/TON23/real_traces/test/geant'

    # Chargement des datasets
    ds_train = input_fn(TRAIN_PATH, shuffle=True).prefetch(tf.data.experimental.AUTOTUNE)
    ds_validation = input_fn(VALIDATION_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = input_fn(TEST_PATH, shuffle=False).prefetch(tf.data.experimental.AUTOTUNE)

    dims = features["dim"]
    iterations = features["iterations"]

    for dim in dims:
        for iteration in iterations:
            path_dim = dim["path_state_dim"]
            link_dim = dim["link_state_dim"]
            queue_dim = dim["queue_state_dim"]

            print(f"\n=== Entraînement avec {dim} et {iteration} iterations ===\n")

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            model = Routenet_Fermi_GRU(
                iterations=iteration,
                path_state_dim=path_dim,
                link_state_dim=link_dim,
                queue_state_dim=queue_dim
            )

            loss_object = tf.keras.losses.MeanAbsolutePercentageError()
            model.compile(loss=loss_object, optimizer=optimizer, run_eagerly=False)

            # Dossier de sauvegarde personnalisé
            file_prefix = f"{iteration}it_{path_dim}p_{link_dim}l_{queue_dim}q"
            ckpt_dir = os.path.join('./ckpts', file_prefix)
            os.makedirs(ckpt_dir, exist_ok=True)

            latest = tf.train.latest_checkpoint(ckpt_dir)
            if latest is not None:
                print("Modèle pré-entraîné trouvé. Chargement...")
                model.load_weights(latest)
            else:
                print("Aucun modèle trouvé. Initialisation aléatoire.")

            filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                mode="min",
                monitor='val_loss',
                save_best_only=False,
                save_weights_only=True,
                save_freq='epoch'
            )

            start_time = time.time()
            history = model.fit(
                ds_train,
                epochs=epoch,
                validation_data=ds_test,
                callbacks=[cp_callback],
                use_multiprocessing=True
            )
            duration = time.time() - start_time

            # Évaluation finale
            print(f"\nÉvaluation sur le test set :")
            model.evaluate(ds_test)

            # Résumé
            print(f"Terminé : {file_prefix} en {duration:.2f} secondes.\n")

            # Sauvegarde des logs
            history_dict = history.history
            json_path = os.path.join(ckpt_dir, f"history_{file_prefix}.json")
            json.dump(history_dict, open(json_path, 'w'))



features = {
    'iterations': [5, 8, 10, 15],
    "dim": [
        {'path_state_dim': 16, 'link_state_dim': 16, 'queue_state_dim': 16},
        {'path_state_dim': 32, 'link_state_dim': 32, 'queue_state_dim': 32},
        {'path_state_dim': 64, 'link_state_dim': 64, 'queue_state_dim': 64},
        {'path_state_dim': 128, 'link_state_dim': 128, 'queue_state_dim': 128},
    ]
}

train_tweak_features(features, epoch=5)
