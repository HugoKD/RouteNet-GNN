## entrainement du modele avec une technique few shot learnning

#Le but de cette section "seed" est d'assurer la reproductibilité de l'entraînement en fixant
# une graine aléatoire pour toutes les opérations utilisant des nombres aléatoires.

seed_value = 69420
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # on utilise le CPU
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value) #on fixe la grainne pour la générationn de nombre aléatoire
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

from data_generator import input_fn

import sys

sys.path.append('../')
from delay_model import RouteNet_Fermi

seed = 0

################################################
################################################

for i in range(5):
    for num_samples in [10000, 5000, 2000, 1000, 500, 100, 50, 25]:
        print("STARTING ITERATION {} WITH NUM SAMPLES {}".format(i, num_samples))

        TRAIN_PATH = '../data/all_mixed/train'
        VALIDATION_PATH = '../data/all_mixed/validation'
        TEST_PATH = '../data/all_mixed/test'

        ckpt_dir = './ckpt_dirs/ckpt_dir_{}_{}'.format(i, num_samples)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model = RouteNet_Fermi()

        loss_object = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(loss=loss_object,
                      optimizer=optimizer,
                      run_eagerly=False)

        filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")
        #Callback pour sauvegarder les meilleurs poids du modèle
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            verbose=1,
            mode="min",
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=True,
            save_freq='epoch')

        #chargement des données d'entraînement et de validation
        ds_train = input_fn(TRAIN_PATH, seed=seed, shuffle=True)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE) #Améliore les performances en préchargeant les données pendant que le modèle s'entraîne.
        ds_train = ds_train.take(num_samples) #echantillonage de ds_train
        ds_train = ds_train.repeat() #fait en sorte que les données d'entraînement soient répétées pendant l'entraînement.

        ds_validation = input_fn(VALIDATION_PATH, shuffle=False)
        ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

        model.load_weights('./initial_weights/initial_weights')

        model.fit(ds_train,
                  epochs=15,
                  steps_per_epoch=2500,
                  validation_data=ds_validation,
                  callbacks=[cp_callback],
                  use_multiprocessing=True)

        seed += 1
