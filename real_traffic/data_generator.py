"""
   Copyright 2021 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import networkx as nx
import numpy as np
import tensorflow as tf

import sys

sys.path.append('../')
from datanetAPI import DatanetAPI  # This API may be different for different versions of the dataset


#TABLE DE ROUTAGE (matrice) : format :  {'AggInfo': {'AvgBw': 0.0, 'PktsGen': 0.0, 'TotalPktsGen': 0.0}, 'Flows': []}
#bande passante, moyenne, nombre de paquets gén par seconde, nombre de paquets gen
#une list de flux contenant : la distribution de paquets ainsi que ses para
#TYhis is a matrix where each cell [i,j] contains aggregated and flow-level
# information about size and time distributions between source i and destination j.


POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO']) #policy for queuing

def generator(data_dir, shuffle):
    try:
        data_dir = data_dir.decode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    it = iter(tool) #iterator
    num_samples = 0
    for sample in it:
        num_samples += 1
        G = nx.DiGraph(sample.get_topology_object()) # info abt topo cf fichier graphe.txt
        T = sample.get_traffic_matrix() #traffic matrix = flow level informations
        R = sample.get_routing_matrix()  # table de routage, ou plutot path pour aller de i à j
        P = sample.get_performance_matrix() # perf
        #à ce point on a encore des matrice en 22*22
        HG = network_to_hypergraph(G=G, R=R, T=T, P=P) # see later, création d'un hypergraphe contenant TOUTES les infos utiles
        ret = hypergraph_to_input_data(HG)
        num_samples += 1 # num sample += 2 ???
        # SKIP SAMPLES WITH ZERO OR NEGATIVE VALUES
        if not all(x > 0 for x in ret[1]):
            continue
        # generator object, we've to iterate with next(generator) to get the value of the next sample ! Input's model
        #ret is a tuple of length 2
        #ret[0] correspond aux informations du network,  dict_keys(['traffic', 'packets', 'length', 'model', 'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor', 'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma', 'capacity', 'queue_size', 'policy', 'priority', 'weight', 'link_t])
        #ret[1] correspond aux targets (ici delay avg par flow),
        #ces delays correpondent aux link to path du graphe (ou au queue to path), pour chaque lien on possède le delay correspond au flow
        #combien de flow on a (ie de chemins (src, destination)) ?
        #pour calculer la loss il suffit de comparer ret[1] avec prediction_delay_real_traces.npy
        yield ret


#transforme un hypergraphe dans un format exploitable par le model RouteNEt
#donne les features (ret[0]) et les targets delay pour chaque flow (ret[1])
def hypergraph_to_input_data(HG):
    n_q = 0
    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(HG.nodes()):
        if entity.startswith('q'):
            mapping[entity] = ('q_{}'.format(n_q))
            n_q += 1
        elif entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = ('l_{}'.format(n_l))
            n_l += 1

    HG = nx.relabel_nodes(HG, mapping)

    link_to_path = []
    queue_to_path = []
    path_to_queue = []
    queue_to_link = []
    path_to_link = []

    for node in HG.nodes:
        in_nodes = [s for s, d in HG.in_edges(node)]
        if node.startswith('q_'):
            path = []
            for n in in_nodes:
                if n.startswith('p_'):
                    path_pos = []
                    for _, d in HG.out_edges(n):
                        if d.startswith('q_'):
                            path_pos.append(d)
                    path.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_queue.append(path)
        elif node.startswith('p_'):
            links = []
            queues = []
            for n in in_nodes:
                if n.startswith('l_'):
                    links.append(int(n.replace('l_', '')))
                elif n.startswith('q_'):
                    queues.append(int(n.replace('q_', '')))
            link_to_path.append(links)
            queue_to_path.append(queues)
        elif node.startswith('l_'):
            queues = []
            paths = []
            for n in in_nodes:
                if n.startswith('q_'):
                    queues.append(int(n.replace('q_', '')))
                elif n.startswith('p_'):
                    path_pos = []
                    for _, d in HG.out_edges(n):
                        if d.startswith('l_'):
                            path_pos.append(d)
                    paths.append([int(n.replace('p_', '')), path_pos.index(node)])
            path_to_link.append(paths)
            queue_to_link.append(queues)
    #print('\n nbr de queue to path (ou link to path) : ', len(queue_to_path))
    return {"traffic": np.expand_dims(list(nx.get_node_attributes(HG, 'traffic').values()), axis=1),
            "packets": np.expand_dims(list(nx.get_node_attributes(HG, 'packets').values()), axis=1),
            "length": list(nx.get_node_attributes(HG, 'length').values()),
            "model": list(nx.get_node_attributes(HG, 'model').values()),
            "eq_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'eq_lambda').values()), axis=1),
            "avg_pkts_lambda": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_pkts_lambda').values()), axis=1),
            "exp_max_factor": np.expand_dims(list(nx.get_node_attributes(HG, 'exp_max_factor').values()), axis=1),
            "pkts_lambda_on": np.expand_dims(list(nx.get_node_attributes(HG, 'pkts_lambda_on').values()), axis=1),
            "avg_t_off": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_t_off').values()), axis=1),
            "avg_t_on": np.expand_dims(list(nx.get_node_attributes(HG, 'avg_t_on').values()), axis=1),
            "ar_a": np.expand_dims(list(nx.get_node_attributes(HG, 'ar_a').values()), axis=1),
            "sigma": np.expand_dims(list(nx.get_node_attributes(HG, 'sigma').values()), axis=1),
            "capacity": np.expand_dims(list(nx.get_node_attributes(HG, 'capacity').values()), axis=1),
            "queue_size": np.expand_dims(list(nx.get_node_attributes(HG, 'queue_size').values()), axis=1),
            "policy": list(nx.get_node_attributes(HG, 'policy').values()),
            "priority": list(nx.get_node_attributes(HG, 'priority').values()),
            "weight": np.expand_dims(list(nx.get_node_attributes(HG, 'weight').values()), axis=1),
            "link_to_path": tf.ragged.constant(link_to_path),
            "queue_to_path": tf.ragged.constant(queue_to_path),
            "queue_to_link": tf.ragged.constant(queue_to_link),
            "path_to_queue": tf.ragged.constant(path_to_queue, ragged_rank=1),
            "path_to_link": tf.ragged.constant(path_to_link, ragged_rank=1)

            }, list(nx.get_node_attributes(HG, 'delay').values())

#transforme une rpz réseau (graphe dirigé) en un hypergraphe contenant toutes les infos utiles sous formes de graphes (liens, chemins, queues)
def network_to_hypergraph(G, R, T, P):
    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if G.has_edge(src, dst):
                    policy = G.nodes[src]['schedulingPolicy'] if 'schedulingPolicy' in G.nodes[src] else 'FIFO'
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 capacity=G.edges[src, dst]['bandwidth'],
                                 policy=np.where(policy == POLICIES)[0][0])
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:

                        time_dist_params = [0] * 8

                        flow = T[src, dst]['Flows'][f_id]
                        model = 0
                        flow['TimeDistParams']['Distribution'] = 'Poisson'

                        if model == 6 and flow['TimeDistParams']['Distribution'] == 'AR1-1': #AR1-1 = Autoregressive -> typique pour modéliser des séries temporelles où la valeur future dépend de la valeur précédente
                            model += 1
                        if 'EqLambda' in flow['TimeDistParams']:
                            time_dist_params[0] = flow['TimeDistParams']['EqLambda']
                        if 'AvgPktsLambda' in flow['TimeDistParams']:
                            time_dist_params[1] = flow['TimeDistParams']['AvgPktsLambda']
                        if 'ExpMaxFactor' in flow['TimeDistParams']:
                            time_dist_params[2] = flow['TimeDistParams']['ExpMaxFactor']
                        if 'PktsLambdaOn' in flow['TimeDistParams']:
                            time_dist_params[3] = flow['TimeDistParams']['PktsLambdaOn']
                        if 'AvgTOff' in flow['TimeDistParams']:
                            time_dist_params[4] = flow['TimeDistParams']['AvgTOff']
                        if 'AvgTOn' in flow['TimeDistParams']:
                            time_dist_params[5] = flow['TimeDistParams']['AvgTOn']
                        if 'AR-a' in flow['TimeDistParams']:
                            time_dist_params[6] = flow['TimeDistParams']['AR-a']
                        if 'sigma' in flow['TimeDistParams']:
                            time_dist_params[7] = flow['TimeDistParams']['sigma']
                        D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),
                                     source=src,
                                     destination=dst,
                                     tos=int(T[src, dst]['Flows'][0]['ToS']),
                                     traffic=T[src, dst]['Flows'][f_id]['AvgBw'],
                                     packets=T[src, dst]['Flows'][f_id]['PktsGen'],
                                     length=len(R[src, dst]) - 1,
                                     model=model,
                                     eq_lambda=time_dist_params[0],
                                     avg_pkts_lambda=time_dist_params[1],
                                     exp_max_factor=time_dist_params[2],
                                     pkts_lambda_on=time_dist_params[3],
                                     avg_t_off=time_dist_params[4],
                                     avg_t_on=time_dist_params[5],
                                     ar_a=time_dist_params[6],
                                     sigma=time_dist_params[7],
                                     delay=P[src, dst]['Flows'][f_id]['AvgDelay'])

                        for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                            # D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                            D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}_{}'.format(src, dst, f_id))
                            D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'l_{}_{}'.format(h_1, h_2))
                            if 'bufferSizes' in G.nodes[h_1]:
                                q_s = str(G.nodes[h_1]['bufferSizes']).split(',')
                            elif 'queueSizes' in G.nodes[h_1]:
                                q_s = [
                                    int(q) * (T[src, dst]['Flows'][f_id]['AvgBw'] / T[src, dst]['Flows'][f_id]['PktsGen'])
                                    for q in str(G.nodes[h_1]['queueSizes']).split(',')]
                            elif 'queueSize' in G.nodes[h_1]:
                                q_s = [
                                    int(q) * (T[src, dst]['Flows'][f_id]['AvgBw'] / T[src, dst]['Flows'][f_id]['PktsGen'])
                                    for q in str(G.nodes[h_1]['queueSize']).split(',')]
                            # policy = G.nodes[h_1]['schedulingPolicy']
                            if 'schedulingWeights' in G.nodes[h_1]:
                                if G.nodes[h_1]['schedulingWeights'] != '-':
                                    q_w = [float(w) for w in str(G.nodes[h_1]['schedulingWeights']).split(',')]
                                    w_sum = sum(q_w)
                                    q_w = [w / w_sum for w in q_w]
                                else:
                                    q_w = ['-']
                            else:
                                q_w = ['-']
                            if 'tosToQoSqueue' in G.nodes[h_1]:
                                q_map = [m.split(',') for m in str(G.nodes[h_1]['tosToQoSqueue']).split(';')]
                            else:
                                q_map = [['0'], ['1'], ['2']]
                            q_n = 0
                            n_queues = G.nodes[h_1]['levelsQoS'] if 'levelsQoS' in G.nodes[h_1] else 1
                            for q in range(n_queues):
                                D_G.add_node('q_{}_{}_{}'.format(h_1, h_2, q),
                                             queue_size=int(q_s[q]),
                                             priority=q_n,
                                             weight=q_w[q] if q_w[0] != '-' else 0)

                                D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'l_{}_{}'.format(h_1, h_2))
                                if str(int(T[src, dst]['Flows'][0]['ToS'])) in q_map[q]:
                                    D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'q_{}_{}_{}'.format(h_1, h_2, q))
                                    D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'p_{}_{}_{}'.format(src, dst, f_id))
                                q_n += 1

    # print([node for node, in_degree in D_G.out_degree() if in_degree == 0])
    D_G.remove_nodes_from([node for node, in_degree in D_G.in_degree() if in_degree == 0])

    return D_G


def input_fn(data_dir, shuffle=False):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[data_dir, shuffle],
                                        output_signature=(
                                            {"traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "length": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "model": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "eq_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "avg_pkts_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "exp_max_factor": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "pkts_lambda_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "avg_t_off": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "avg_t_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "ar_a": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "sigma": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "queue_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "policy": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "priority": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "weight": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "link_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "queue_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "queue_to_link": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "path_to_queue": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32,
                                                                                  ragged_rank=1),
                                             "path_to_link": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32,
                                                                                 ragged_rank=1)
                                             }
                                            , tf.TensorSpec(shape=None, dtype=tf.float32)
                                        ))

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
