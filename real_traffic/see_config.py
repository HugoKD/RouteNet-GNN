'''
########################## CONTENU DES DONNÉES POUR TESTER ##########################

Dans le dossier test/ :
- 2 types de réseaux (ex: geant, germany50)
- Pour chacun → un ensemble de features (22) + 1 target (retard en s)

features = [
  'traffic', 'packets', 'length', 'model', 'eq_lambda', 'avg_pkts_lambda',
  'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma',
  'capacity', 'queue_size', 'policy', 'priority', 'weight',
  'link_to_path', 'queue_to_path', 'queue_to_link', 'path_to_queue', 'path_to_link'
]

→ len(features) = 22  
→ target = vecteur de taille 409 ou 400 (nombre de flux pour chaque réseau)
→ Les valeurs sont des délais (10^-6 à 10^-3 secondes)

######################################################################
########################## ANALYSE DES FEATURES ######################
######################################################################

# Fichier traffic.txt : contient les flux générés (source/dest + intensité)
- TOS = type of service (définit la priorité du flux via QoS)
  - 0 → faible prio, 1 → moyenne, 2 → élevée
  - Ex: VoIP (latence faible) → prio élevée, téléchargement → prio basse
- Les files d’attente (queues) gèrent ça via les politiques (FIFO, SP, WFQ...)

# Données traitées :
- shape des features = (409, ...) → chaque ligne représente un flux
- Exemple : 409 flux pour le réseau GEANT (22x22 = 484 paires possibles)

#### Détails :
traffic             → intensité du flux
packets             → nbr de paquets du flux
length              → nbr de liens traversés
model               → type de distribution de temps (Poisson, OnOff, etc.)
avg_pkts_lambda     → λ moyen (souvent nul avec Poisson)
exp_max_factor      → utilisé si modèle exponentiel (sinon 0)
pkts_lambda_on      → λ en phase ON
avg_t_off / avg_t_on→ durées OFF/ON (nulles pour Poisson)
ar_a / sigma        → pour auto-régression et variance (nulles sauf cas spé)
capacity            → capacité des liens (71,)
queue_size          → tailles des buffers (111,), valeurs = [8k, 16k, 32k, 64k]
policy              → politique d’ordonnancement (FIFO, SP, DRR, WFQ)
priority / weight   → params des queues (111,)
link_to_path        → pour chaque flux, liens parcourus
queue_to_path       → idem pour queues
queue_to_link       → pour chaque lien, quelles queues l'alimentent
path_to_queue       → pour chaque queue, quels flux l’utilisent (et où)
path_to_link        → idem pour liens

Un flux traverse autant de queues que de liens
→ Même si un lien peut avoir plusieurs queues, un flux ne choisit qu’une seule par lien.

network, chaque lien est traversé par un nombre variable de chemin -> dou la dim 2 = None , dim 3 = 2 car on a (indice du lien, position du lien)
Par exemple : path_to_lin[10]=ensemble des flux passant par le lien 10 : tf.Tensor(
[[  9   1] #le premier element est bien la numérotation du flux -> max 408, min = 0,
 [ 10   1] #le deuxieme element est bien la position du lien dans le flux ce qui est logique avec le fait qu'au max un flux à une longueur de 5 liens (max = 4, min = 0)
 [ 65   1]
 [ 66   1]
 [ 83   0]
 [ 84   0]
 [124   1]
 [160   3]
 [178   2]
 [179   2]
 [196   1]
 [250   2]
 [271   1]
 [290   2]
 [308   2]
 [309   2]
 [347   1]
 [348   1]
 [364   2]
 [365   2]
 [381   2]
 [400   2]], shape=(22, 2), dtype=int32)
 
RQ : 
- Un flux passe par le meme nombre de queues que de liens. S'il passe par 3 liens, il passe aussi par 3 queues. 
- Il ne peut pas se spliter en plusieurs queues pour un
- Même lien (ce qui est theoriquement possible puisque un lien peut avoir plusieurs queues (cf queue_to_link))
######################################################################
####################### TABLE DE ROUTAGE #############################
######################################################################

- Fichier de routage → représente l’ensemble des paths entre [src, dst]
- Ex : 22 nœuds → 22x22 = 484 combinaisons
→ La table définit, pour chaque i,j, le port à utiliser pour atteindre j depuis i
→ Valeur -1 si pas de chemin possible

Exploité par DatanetAPI pour construire les matrices de features
Ensuite a partir de cette table, on peut connaitre quel noeud atteindre en etant au current node , et ensuite quel chemin prendre pour aller de i à j


######################################################################
######################### TRAFIC MODELS ##############################
######################################################################

Type de modèles (TimeDistribution) :
0 = Exponential
1 = Déterministe
2 = Uniforme
3 = Normale
4 = ON-OFF
5 = PPBP
6 = TRACE
7 = EXTERNAL_PY

→ Chaque modèle impacte comment les paquets sont générés dans le temps.

######################################################################
############## Lien entre état des files et état des liens ###########
######################################################################

- Un lien = ensemble de queues → l'état du lien est une fonction de ces queues
- Un flux = ensemble de queues + liens → son état dépend de toute sa trajectoire

###########################################################################################
######### PERFORMANCES simulationResults.txt (target = delay/jitter/losses) ###############
###########################################################################################

Fichier simulationResults.txt :
→ Partie globale : global_packets (nbr de paquets dans le réseau), global_losses, global_delay :: (partie avant |)
→ Partie flowResults : par paire source/destination :: (partie après le |) qui a aussi une longueur de 484

    - AvgDelay, p10/p50/p90, jitter, PktsDrop...
    - Aussi disponible : par flux individuel

Indice (i, j) dans la matrice = index 22*i + j

Est-ce que les 484 values de la matrice représentent les delays de chaque flux traversant un nœud ?
→ Possible, mais le mapping est à reconstruire via routing info


######################################################################
######################### traffic.txt ################################
######################################################################

- Deux sections : avant `|` et après
- On extrait : maxAvgLambda + liste des flux (shape = 484)

→ Aligné avec les matrices de performance, 1 entrée = 1 flux

Attention : les flux finaux considérés sont 400 ou 409 → car certains flux (i,j) n'ont peut-être pas été retenus

######################################################################
######################### linkUsage.txt ##############################
######################################################################

Contient : pour chaque config réseau :
- infos i,j séparées par `;`, et par `:` dans chaque couple
- Soit -1 si pas de lien
- Soit : utilization, loss, avg_pkt_size + listes des stats des queues associées

Utilisé pour créer `portStat`, structure complète pour chaque lien :
- Dictionnaire avec :
    → [link_utilization, losses, avg_pkt_size]
    → [liste de stats par queue]
    
############# TRAFFIC ET RESULTS ####################################

In fine ces deux informations t et r servent à créer deux matrices respectivemnt une matrice de traffic et une de performance, ces matrices 
sont encore une fois de la même dimension qu'auparavant c'est à dire 22*22 -> 22 comme le nombre de noeuds dans le réseau

######################################################################
############### QUESTIONS SUR L’APPRENTISSAGE ########################
######################################################################

Comment se fait l’entraînement ?
- On entraîne l’ensemble : embeddings + GRU/LSTM + readout
- B.P. (backpropagation) s’applique de bout en bout

############################# REMARQUES ##############################

Modèle testé sur 4 scénarios :
- Modèles de trafic non poissonien
- Routing constant entre train/test
- Routing différent
- Simulation de pannes de lien (topo acyclique)

Objectif = tester robustesse + généralisation

Pourquoi utiliser le taux de charge (load) plutôt que capacity seule ?
→ Car plus robuste au scaling sur les réseaux plus larges

######################################################################
########################## TODO (Soutenance) #########################
######################################################################

- Tester inférence sur réseaux 30× plus grands que ceux d’entraînement
- Tester sur petits datasets
- Approfondir les maths des GNN (théorie + implémentation)
- Comprendre chaque repo
- Analyser en quoi cela s’applique aux réseaux 5G/6G
- Pourquoi les checkpoints classiques ne suffisaient pas ?
- Approfondir LSTM
- Ajouter mémoire plus longue : T-2, T-3 ?
- Relativiser intérêt du LSTM ici : flux très courts (5 étapes max)
→ Donc historique limité = moins d’impact que sur des séquences longues

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

