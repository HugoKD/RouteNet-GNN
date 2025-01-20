'''import pandas as pd
import os

path_dir= f'../data/TON23/real_traces/test/test/results_geant_1000_0_1'

lst_dir = os.listdir(path_dir)
dfs = {}
for tm in lst_dir :
    print(tm)
    dfs[tm] = pd.read_csv(path_dir+ '/' + tm, delimiter=';')  # if it's tab-delimited

print(dfs[lst_dir[2]].head())'''

''' 
Dans le repo test on a 2 types de network. 
Pour chacun on a ses features et sa targets 
on a comme feature pour chaque : 
features = ['traffic', 'packets', 'length', 'model', 'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma', 'capacity', 'queue_size', 'policy', 'priority', 'weight', 'link_to_path', 'queue_to_path', 'queue_to_link', 'path_to_queue', 'path_to_link']
avec len(features) = 22
Aussi on a la target un tensor de shape (409,) pour la premiere instance et (400,) pour la deuxième
les targets contiennent des valeurs variants de l'ordre de 10e-6 à 10e-3 -> reatrd exprimé en seconde 

################################################################################################################################
########################################################## Features Analysis ###################################################
################################################################################################################################

DataAPiNet ne s interesse que au dossier tar 
traffic (409, 1)
packets (409, 1)
length (409,)
model (409,)
eq_lambda (409, 1)
avg_pkts_lambda (409, 1)
exp_max_factor (409, 1)
pkts_lambda_on (409, 1)
avg_t_off (409, 1)
avg_t_on (409, 1)
ar_a (409, 1)
sigma (409, 1)
capacity (71, 1)
queue_size (111, 1)
policy (71,)
priority (111,)
weight (111, 1)
link_to_path (409, 1)
queue_to_path (409, 1)
queue_to_link (71, 1)
path_to_queue (111, None, 2)
path_to_link (71, None, 2)

Le model représente un type de modèle ou de distribution de temps utilisé pour simuler le comportement d'UN flux.
En tout on a 6 type de traffic model :  Poisson, On-Off, Constant Bitrate, Autocorrelated Exponentials, Modulated Exponentials and all models mixed.

Table de routage en input (routing-giant .. par ex), correspond à l ensemble des [src,dst]. Dans l'exemple test/giant on a 22*22 elements = 486
La fonction crée une matrice de routage où chaque cellule R[i][j] représente le port à utiliser pour que le nœud i puisse atteindre le nœud j. 
Si aucune information n'est trouvée pour un chemin spécifique, la valeur restera -1 
(ce qui peut signifier qu'il n'y a pas de chemin défini entre ces deux nœuds dans le fichier de routage).

les files graphes sont des fichiers de topologie au format GML (Graph Modelling Language)networkx et sont retourner comme des objects networkx

le fichier simulationresults gloablement partage les données suivantes : 

global_packets (nbr de paquets dans le réseau), global_losses, global_delay (partie avant |)
et _flowresults_line (partie après le |) qui a aussi une longueur de 484

Meme chose pour traffic.txt : 
2 types d'information (avt | et apres) :
 - maxAvgLambda which likely represents the maximum average traffic arrival rate across all flows or paths in the network
 - traffic list 

In fine ces deux informations t et r servent à créer deux matrices respectivemnt une matrice de traffic et une de performance, ces matrices sont encore 
une fois de la même dimension qu'auparavant c'est à dire 22*22 -> 22 comme le nombre de noeuds dans le réseau 

Premier constat : le nombre de noeuds ne correspond pas à la shape de prédiction (par exemple 486 vs 400 ou 409)
(bien une information at flow scale, ?)

La matrice de perf est tq : pour chaque paire source/destination : 
on a une info aggregate information (Données agrégées pour toutes les connexions (ou flux) entre les deux nœuds) et flows (Liste contenant des métriques détaillées pour chaque flux individuel entre les deux nœuds.)
PktsDrop: Nombre total de paquets perdus.
AvgDelay: Délai moyen des paquets
AvgLnDelay: Délai logarithmique moyen (souvent utilisé pour lisser les valeurs extrêmes ou représenter des échelles).
p10, p20, p50, p80, p90: Percentiles des délais des paquets (par exemple, p50 est la médiane).
Jitter: Variation dans les délais des paquets (indicateur de qualité pour des applications sensibles comme la voix sur IP).
2. Flows: Détails des flux
Chaque flux représente une connexion individuelle entre la source et la destination :
Les métriques sont similaires à celles d'AggInfo, mais spécifiques à chaque flux.

 '''
import os
import tensorflow as tf
from data_generator import input_fn

import sys

#sys.path.append('../../')
from delay_model import RouteNet_Fermi

TRAIN_PATH = f'../data/TON23/real_traces/test/test'
VALIDATION_PATH = f'/data/TON23/real_traces'
TEST_PATH = f'/data/TON23/real_traces'

ds_train = input_fn(TRAIN_PATH, shuffle=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_validation = input_fn(VALIDATION_PATH, shuffle=False)
ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

for data_batch in ds_train.take(2):
    print(data_batch[0].keys())
    for c in data_batch[0].keys():
        print(c, data_batch[0][c].shape)
    print(data_batch[0]["traffic"])
