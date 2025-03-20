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

Fichier traffic.txt : définit les flux de trafic qui vont traverser le réseau pendant la simulation
RQ : le TOS définit la priorité du flux en fonction de la qualité de service (QoS).
    0 → Priorité faible.
    1 → Priorité moyenne.
    2 → Priorité élevée.
En effet, différent type de service (streaming, telechargement de fichier) ont des prioritées differentes. Par ex une visio (latence faible) aura un prio plus forte qu'un 
téléchargement d'un fichier. Les queue sont ceux qui gèrent ce type of service avec la QOS (quality of service) et leur différent type de polotiques (fifo, SP (les paquets 
prioritaires passent toujours en premiers),...). Par exemple un paquet TOS = 2, sera prioritaire avec une queue SP, aura une bande passante plus grande avec une queue WFQ
et ne sera pas prioritaire avec une queue FIFO.

DataAPiNet ne s interesse que au dossier tar 
Quand on lit les informations fournies par notre dataset on a bien des matrices shape = 409 ou 400, pourquoi 409 then = nbr de flux !
Cependant ce n'est pas comme cela qu'elles sont de base (par exemple, performance matrix.shape = 22*22)


#On a 409 flux qui traversent le réseau :
traffic (409, 1) 
packets (409, 1) #nbr de paquets par flow ?
length (409,) #nbr de liens traversés par le flux -> lentgh[i] = j => le flux i passe par j liens -> à comparer avec d'autre feature comme link_to_path ?
model (409,) #type de modèle/distribution de temps utilisé pour simuler le comportement d'UN flux Parmis 
avg_pkts_lambda (409, 1) #que des 0, Moyenne du taux d'arrivée des paquets (lambda)	 ### POURQUOI EST CE QUE C'EST NUL ? ######
exp_max_factor (409, 1)#que des 0,Facteur de maximisation exponentielle, peut etre 0 si non utilisé
pkts_lambda_on (409, 1)#que des 0, Taux d’arrivée des paquets en phase "ON" (ex: ON-OFF)	
avg_t_off (409, 1)#que des 0, Durée moyenne des périodes OFF (ex: ON-OFF)	-> normale pour expo
avg_t_on (409, 1)#que des 0,Durée moyenne des périodes ON (ex: ON-OFF)	-> normale pour expo
ar_a (409, 1)#que des 0, facteur d'auto regression,non utilisé avec des expo (entre autre, les paquets sont non corrélés)
sigma (409, 1)#que des 0, utilisé pour les modeles avec des normales 
capacity (71, 1) #bw
queue_size (111, 1) -> max = 64k, min = 8k (8,16,32,64bits), plusieurs politiqes par link si SP, WFQ ou DRR. Max 3 queues per port 
policy (71,) #Même nombre de liens que de policy -> 71, seulement 4 policies (FIFO, SP, WFQ,DRR) 'POLITIQUE DES QUEUS'
priority (111,) #même nombre que le nombre de queues
weight (111, 1)#même nombre que le nombre de queues
link_to_path (409, 1) #Pour chaque chemin on a sa composition en terme de liens (conformémement au feature length)
queue_to_path (409, 1) # de même pour les queues 
queue_to_link (71, 1) #Pour chaque lien, l'ensemble des queues qui sont comme une 'source' ce lien 
path_to_queue (111, None, 2) #même chose que pour path_to_link
path_to_link (71, None, 2) # path_to_link[i] correspond à tous les chemins qui utilisent le lien l_i et leur position dans ces chemins. -> il y a 71 liens dans le 
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



Un flux passe par le meme nombre de queue que de lien. S'il passe par 3 liens, il passe aussi par 3 queues. Il ne peut pas se spliter en plusieurs queues pour un 
meme lien (ce qui est theoriquement possible puisque un lien peut avoir plusieurs queues (cf queue_to_link)) 

###########################################################################
A partir des tables de routage, datanet api créer une matrice de routage ou chaque element i,j correspond au chemin pour aller de i à j (reellement chemin à travers
les noeuds).


Les différents types de modeles (=timeDist) : 
EXPONENTIAL_T = 0
DETERMINISTIC_T = 1
UNIFORM_T = 2
NORMAL_T = 3
ONOFF_T = 4
PPBP_T = 5
TRACE_T = 6
EXTERNAL_PY_T = 7

L'état d'un lien à la fin dépend de l'état de chaque queue qui sont à l'entrée de ce lien (ie qui injecte du traffic dans le lien l (port de sortie vers l))
L'état d'un flow dépend de l'ensemble des queues et de liens le composant 

Table de routage en input (routing-giant .. par ex), correspond à l ensemble des [src,dst]. Dans l'exemple test/giant on a 22*22 elements = 486 pour les 22 nodes = network size 
C'est donc celle ci qui donne l'information sur l'ensemble des flows qui vont traverser le réseau ?
La fonction crée une matrice de routage où chaque cellule R[i][j] représente le port à utiliser pour que le nœud i puisse atteindre le nœud j. Si aucune information n'est trouvée pour un chemin spécifique, la valeur restera -1 
(ce qui peut signifier qu'il n'y a pas de chemin défini entre ces deux nœuds dans le fichier de routage).
Ensuite a partir de cette table, on peut connaitre quel noeud atteindre en etant au current node , et ensuite quel chemin prendre pour aller de i à j 

###########################################################################
les files graphes sont des fichiers de topologie au format GML (Graph Modelling Language)networkx et sont retourner comme des objects networkx

###########################################################################
le fichier simulationresults gloablement partage les données suivantes : 

global_packets (nbr de paquets dans le réseau), global_losses, global_delay (partie avant |)
et _flowresults_line (partie après le |) qui a aussi une longueur de 484

pour les données flowResults : 

La matrice de perf source -> destination est tq : pour chaque paire source/destination : 
on a une info aggregate information (Données agrégées pour toutes les connexions (ou flux) entre les deux nœuds) et flows (Liste contenant des métriques détaillées pour chaque flux individuel entre les deux nœuds.)
PktsDrop: Nombre total de paquets perdus.

Chaque (i,j) contient differentes informations tq : 
- AvgDelay: Délai moyen des paquets
- AvgLnDelay: Délai logarithmique moyen (souvent utilisé pour lisser les valeurs extrêmes ou représenter des échelles).
- p10, p20, p50, p80, p90: Percentiles des délais des paquets (par exemple, p50 est la médiane).
- Jitter: Variation dans les délais des paquets (indicateur de qualité pour des applications sensibles comme la voix sur IP).

2. Flows: Détails des flux
Chaque flux représente une connexion individuelle entre la source et la destination :
Les métriques sont similaires à celles d'AggInfo, mais spécifiques à chaque flux.


La question que je me pose : est ce que chaque element de la matrice de performance correpond au delay au niveau du noeud pour chaque flow traversant ce meme noeud ? Au maximum on aurait donc 11 flow qui traverse un noeud 
Chaque element (i,j) est situé à l'indice "22*i + j" 


###########################################################################
Meme chose pour traffic.txt : 
2 types d'information (avt | et apres) :
 - maxAvgLambda which likely represents the maximum average traffic arrival rate across all flows or paths in the network
 - traffic list 

In fine ces deux informations t et r servent à créer deux matrices respectivemnt une matrice de traffic et une de performance, ces matrices sont encore 
une fois de la même dimension qu'auparavant c'est à dire 22*22 -> 22 comme le nombre de noeuds dans le réseau 

Premier constat : le nombre de noeuds ne correspond pas à la shape de prédiction (par exemple 486 vs 400 ou 409)
-> 400 et 409 représente chaque flow !


dans le fichier linkUsage, on a pour chaque ligne une configuration possible d'un réseau, chaque couple noeud i, noeud j est séparé par ';' et chaque information dans ce couple est séparée par ':'
Pou rchaque couple on a soit -1 si le couple n'est pas relié ou soit le premier element séparé par ':' est [utilization, losses, avgpacketsize] et les autres sont des inforamtions sur la gestion des files d'attentes qui composent
le noeud (un noeud peut avoir differnetes filles d attente : utilization""losses" "avgPortOccupancy" "maxQueueOccupancy" "avgPacketSize"
In fine cela sert à créer portStat, pour chaque link (ie pour chaque arrete i,j) on va avoir un dictionnaire ou les 3 premiers elements correpondes aux infs des liens et l'auter correpsond à une liste d'informations sur les queues

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

