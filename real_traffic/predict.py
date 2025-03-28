'''
Faire une prédiction en prenant les meilleurs poids de modele routenet après entraienement

Ici les real traces contiennent des topologies qui ont reellement erxistées
Abilene : Une ancienne infrastructure de recherche utilisée aux États-Unis
GEANT : Une infrastructure réseau européenne interconnectant des institutions de recherche
Germany50 et Nobel : Des topologies de référence utilisées dans la recherche en réseau
les fichiers graphs sont au format graphML (XLM based file format for grpahs)

Pour chaque network possible dans notre DS on a un dossier result_nom_....tar.gz qui contient les config du network
Dans ce dossier on a :
- Input File : associé une topologie à un routage
- des info sur le link usage
- des infos sur le simulation results
- sur stability
- traffic
- avg delay

On tets la predict sur deux configurations possibles dans le directory test/test

La prediction en sortie du train est la concatenation de tous les networks. Par exemple si j'ai deux networks avec des targets de longueur respective 400
et 409, ma prediction finale sera de 809
'''

'''
##############################################
###### Faire une prédiction après train #####
##############################################

Objectif : Utiliser les meilleurs poids entraînés du modèle RouteNet-Fermi (repo ckpt) pour prédire les délais sur les vraies topologies

##############################################
####  Données : Real Traces (test set) #####
##############################################

Les topologies présentes sont inspirées de réseaux **réels ou standards de recherche** :

- `Abilene` : Ancien réseau de recherche US (Internet2)
- `GEANT`   : Réseau académique européen
- `Germany50` & `Nobel` : Topologies classiques utilisées en recherche (librairies comme SNDlib)
- Les graphes sont encodés au format `.graphml` (basé XML pour les graphes)

##############################################
#### Contenu de chaque dossier tar.gz #####
##############################################

Chaque fichier `results_<network>_...tar.gz` contient une configuration complète du réseau et ses métriques associées :
- `input.txt`       : associe la topologie à un routage spécifique
- `linkUsage.txt`   : stats sur l’utilisation des liens (utilisation, pertes, taille moyenne des paquets)
- `simulationResults.txt` : performances globales et par flux
- `stability.txt`   : mesure de convergence/perturbations
- `traffic.txt`     : détails sur les flux simulés
- `avgDelay.txt`    : résumé du délai moyen (optionnel)

Ces fichiers sont utilisés pour générer les features en entrée du modèle + la ground truth

##############################################
#### Phase de test (inférence) ###########
##############################################

Dans le répertoire `test/test`, on a :
- Deux instances (réseaux) testées séparément
- Chaque réseau possède un nombre de flux différent :
  → Exemple : 409 flux pour GEANT, 400 flux pour Nobel

Lorsqu’on applique le modèle sur l’ensemble du répertoire `test/test`, la prédiction est une **concaténation** des résultats sur chaque réseau

Exemple :
- test 1 : 409 flux
- test 2 : 400 flux
→ prédiction finale = vecteur de taille 809 (409 + 400)

À comparer avec la target globale (target = délai réel mesuré pour chaque flux)
'''




results = {
    'real_traces/test/test' : [1.755254983338671,5.607331761632386,809],
    'real_traces/test/abilene' : [3.128438592539247,4.726860964017293,131472,],
    'real_traces/test/germany50' :[8.760057463155169,14.079925831144418,357995,],
    'real_traces/test/geant' :[2.290086764395679,1.8193376845215492,315044,],
    'real_traces/test/nobel': [9.868019183485151, 10.40961680566105, 34030, ]
}
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import numpy as np
import tensorflow as tf
from data_generator import input_fn # Data generator utilise datanet API

import sys
from delay_model_LSTM import RouteNet_Fermi


TEST_PATH = f'../data/TON23/real_traces/test/geant'

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = RouteNet_Fermi()

loss_object = tf.keras.losses.MeanAbsolutePercentageError() # MAPE

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False) #Pas d'early stop

best = None
best_mre = float('inf') #  Mean Relative Error

ckpt_dir = f'./ckpt_dir_LSTM'

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
length = []
y_true = []

for x, y in ds_test:
    y_true.extend(y.numpy())

predictions = model.predict(ds_test, verbose=1)
predictions = np.squeeze(predictions)


pred = predictions.tolist()


def calculate_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    # Avoid division by zero by filtering out zeros in the actual values
    non_zero_actual = actual != 0
    actual = actual[non_zero_actual]
    predicted = predicted[non_zero_actual]

    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape

print('le score MAPE est : ', calculate_mape(y_true, pred)) # prediction sur l ensemble des link to path
np.save(f'predictions_delay_real_traces.npy', np.squeeze(predictions))
print(np.squeeze(predictions))
print(np.squeeze(predictions).shape)