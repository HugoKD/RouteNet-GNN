import tensorflow as tf
# Les fonctions utiliser dans le papier sont des GRUs, voir l'auter implémentation pour utiliser des LSTM -> meilleurs resultats
#  sequential model with an RNN (Figure 4). Particularly,we choose a Gated Recurrent Unit (GRU).
"""
Fonctionnement de l'algo : 
On initialise tous les q,l,f en encodant les infos de base dans un vecteur dense
Etape 1 : Message passing (en T steps)
- pour chaque flow, on vient encoder les etats des liens que le composent en prenant compte l'etat du lien au global ainsi que l'état de q
(on rappelle uin flow est composé d'un ensemble de liens f_i = {(q_(i,1),l_(i,1)....}), on itere sur chaque couple (q,l) E f 
- on genere un message propre aux q de f qui traduis cet embedding de f_l_t
- pour chaque q dans le réseau : Le nouveau état de q est obtenu en prennant en compte tous les états de f_q_t qui traverse q (tous les liens qui traversent q)
et puis on update l'état q en prennant en compte ce nouveau calcul avec l'état de q auparavent 
- on fait la meme chose pour les liens, on ititialise une fonction, pour tous les q composants l on aggrege et update
Etape 2 : Partie Readout 
- on va venir calculer et predire les mesure pertinentes en utilisant les états de flows
- Pour chaque couple (q,l) composant le flow f, on va venir calculer via une fonction readout le delay "instantané" du lien, via 
la valeur de l'emdedding du lien de ce flow au temps t = T (final). On divise cela par la capacité du lien pour obtenir le delay sur le lien.(bande passante du lien)
La prédiction glabale est la somme des delays sur les liens de chaque flow. (en tout on calcule le delay de transmission et du queuing, en gros 
combien de temps on perd à etre dans la file d'attente ainsi que le delai pour transmettre un paquet d'un buffer à un autre)   
On fait la meme avec le jitter along the flow)
La mesure paquet loss est calculé via le readout de l'embedding du flow f au temps t = T.

En gros link/queue to path -> path to queue -> queue to link 

parametre à tunner : nombre de step dans le message passing, nombre de neurones dans les GRU, voir si LSTM mieux que GRU  et les espaces latents
"""
class RouteNet_Fermi(tf.keras.Model):
    def __init__(self):
        super(RouteNet_Fermi, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_num_models = 7

        self.num_policies = 4
        self.max_num_queues = 3

        self.iterations = 8 #nombre de step dans le message passing
        self.path_state_dim = 32 # 32 = dimension de l'embedding du chemin (source, destination)
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.z_score = {'traffic': [1385.4058837890625, 859.8118896484375], #normalisation des entrées
                        'packets': [1.4015231132507324, 0.8932565450668335],
                        'eq_lambda': [1350.97119140625, 858.316162109375],
                        'avg_pkts_lambda': [0.9117304086685181, 0.9723503589630127],
                        'exp_max_factor': [6.663637638092041, 4.715115070343018],
                        'pkts_lambda_on': [0.9116322994232178, 1.651275396347046],
                        'avg_t_off': [1.6649284362792969, 2.356407403945923],
                        'avg_t_on': [1.6649284362792969, 2.356407403945923], 'ar_a': [0.0, 1.0], 'sigma': [0.0, 1.0],
                        'capacity': [27611.091796875, 20090.62109375], 'queue_size': [30259.10546875, 21410.095703125]}

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=10 + self.max_num_models),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu)
        ])

        self.queue_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.max_num_queues + 2),
            tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu), #rpz denses des queues
            tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu)
        ])

        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.num_policies + 1),
            tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu)
        ])
        #fn readout pour lire chaque chemin qui sert à calculer la sortie finale du modele (delai sur chemin par ex)
        # un flow suit un path (source -> destination), un flow est un ensemble f_i = {(q_(i,1),l_(i,1)....}
        self.readout_path = tf.keras.Sequential([ # pas un gru
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(int(self.link_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.path_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")

    @tf.function
    def call(self, inputs):
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model'] #voir ce que ca represente
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        pkts_lambda_on = inputs['pkts_lambda_on']
        avg_t_off = inputs['avg_t_off']
        avg_t_on = inputs['avg_t_on']
        ar_a = inputs['ar_a']
        sigma = inputs['sigma']

        capacity = inputs['capacity']
        policy = tf.one_hot(inputs['policy'], self.num_policies)

        queue_size = inputs['queue_size']
        priority = tf.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']

        queue_to_path = inputs['queue_to_path']
        link_to_path = inputs['link_to_path']
        path_to_link = inputs['path_to_link']
        path_to_queue = inputs['path_to_queue']
        queue_to_link = inputs['queue_to_link']

        #calcul du taux de charge
        # au début le taux est nul ?
        # see what s traffic, peut être un tenseur représentant le trafic généré pour chaque chemin shape = (num_paths,) ?
        # path_to_link : Une matrice de mappage qui indique quels chemins passent par quels liens dans le réseau.
        # Elle est probablement de dimension (num_paths, num_links, 2)
        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        #calcule la somme totale du traffic pour chaque lien
        #capacity surement un vecteur representant la capacité de chaque lien
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets #taille moyen des paquets

        # Initialize the initial hidden state for links
        #information flow level -> traffic sur le chemin
        path_state = self.path_embedding(tf.concat(
            [(traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
             (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
             tf.one_hot(model, self.max_num_models), #encode le traffic utilisé (poisson, constant bit rate, ... Chaque chemin peut avoit son propre traffic (max dans le network  self.max_num_models)
             (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
             (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
             (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
             (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
             (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
             (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
             (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
             (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1]], axis=1))

        # Initialize the initial hidden state for paths
        # l etat du lien dépend de son chargmenet et de sa politique (FIFO, SP, WFQ, DRR ..)
        link_state = self.link_embedding(tf.concat([load, policy], axis=1))

        # Initialize the initial hidden state for paths
        # si DRRR ou WFQ alors on précise le poids, par exemple une file avec un poids de 2
        # aura 2fois plus de bande passante que celle avec un poids de 1
        queue_state = self.queue_embedding(
            tf.concat([(queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
                       priority, weight], axis=1))

        # Iterate t times doing the message passing
        for it in range(self.iterations):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)
            previous_path_state = path_state

            path_state_sequence, path_state = path_update_rnn(tf.concat([queue_gather, link_gather], axis=2),
                                                              initial_state=path_state)

            path_state_sequence = tf.concat([tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1)

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_link)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        #Readout = flow level prediction
        #capacité pour chaque lien d'un chemin donnée grace au mappage link to path
        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor) #predit l'occupation des files pour chaque chemin
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather,
                                         axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        return queue_delay + trans_delay #retard est du à la durée de transmission sur le chemin + la durée de passage dans le queue
