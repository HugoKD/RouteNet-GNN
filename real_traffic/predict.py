# Faire une prédiction en prenant les meilleurs poids de modele routenet après entraienement
#ici les real traces contiennent des topologies qui ont reellement erxistées
#Abilene : Une ancienne infrastructure de recherche utilisée aux États-Unis
#GEANT : Une infrastructure réseau européenne interconnectant des institutions de recherche
#Germany50 et Nobel : Des topologies de référence utilisées dans la recherche en réseau
#les fichiers graphs sont au format graphML (XLM based file format for grpahs)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import numpy as np
import tensorflow as tf
from data_generator import input_fn # Data generator utilise datanet API

import sys

from delay_model import RouteNet_Fermi

TEST_PATH = f'../data/TON23/real_traces/test/test'

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = RouteNet_Fermi()

loss_object = tf.keras.losses.MeanAbsolutePercentageError() # MAPE

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False) #Pas d'early stop

best = None
best_mre = float('inf') #  Mean Relative Error

ckpt_dir = f'./ckpt_dir'

#deux fichiers, l'index -> cxontient l'index des vairbales du model
# .datra-0000-of ... contient les valeurs reeelles des poids et des biais
#on prend le meilleur
for f in os.listdir(ckpt_dir): #Le score MRE est doné dans le nom du checkpoint
    if os.path.isfile(os.path.join(ckpt_dir, f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = f.replace('.index', '')
                best = best.replace('.data', '')
                best = best.replace('-00000-of-00001', '')
                best_mre = mre

print("BEST CHECKOINT FOUND FOR: {}".format(best))
model.load_weights(os.path.join(ckpt_dir, best))

ds_test = input_fn(TEST_PATH, shuffle=False)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

predictions = model.predict(ds_test, verbose=1)

np.save(f'predictions_delay_real_traces.npy', np.squeeze(predictions))
