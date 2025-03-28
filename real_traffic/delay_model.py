import tensorflow as tf


class RouteNet_Fermi(tf.keras.Model):
    def __init__(self , iterations = 8 ,path_state_dim = 32, link_state_dim = 32,queue_state_dim=32):
        super(RouteNet_Fermi,self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_num_models = 7

        self.num_policies = 4
        self.max_num_queues = 3

        self.iterations = iterations
        self.path_state_dim = path_state_dim
        self.link_state_dim = link_state_dim
        self.queue_state_dim = queue_state_dim

        self.z_score = {'traffic': [1385.4058837890625, 859.8118896484375],
                        'packets': [1.4015231132507324, 0.8932565450668335],
                        'eq_lambda': [1350.97119140625, 858.316162109375],
                        'avg_pkts_lambda': [0.9117304086685181, 0.9723503589630127],
                        'exp_max_factor': [6.663637638092041, 4.715115070343018],
                        'pkts_lambda_on': [0.9116322994232178, 1.651275396347046],
                        'avg_t_off': [1.6649284362792969, 2.356407403945923],
                        'avg_t_on': [1.6649284362792969, 2.356407403945923], 'ar_a': [0.0, 1.0], 'sigma': [0.0, 1.0],
                        'capacity': [27611.091796875, 20090.62109375], 'queue_size': [30259.10546875, 21410.095703125]}

        # GRU Cells used in the Message Passing step
        #Translate the dependance of (link, queue) throught each flow
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        #10 car 10 caracteristiques + type_model one hot encodé
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=10 + self.max_num_models),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.path_state_dim, activation=tf.keras.activations.relu)
        ])

        self.queue_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.max_num_queues + 2), # Shape = (None, input_dim), en gros la dimension 1 n'est pas spé, elle correspond au 'batch_size',
            #ici au nombre de queues à considérer
            tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.queue_state_dim, activation=tf.keras.activations.relu)
        ])

        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.num_policies + 1),
            tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(self.link_state_dim, activation=tf.keras.activations.relu)
        ])

        # Ce bloc correspond à la fonction de lecture (readout) appliquée aux chemins (paths)
        # Elle prend en entrée une séquence de vecteurs latents de dimension `path_state_dim`
        # - La dimension `None` correspond au nombre d'étapes du chemin (i.e., le nombre de liens traversés), variable selon les flux
        # - La première dimension (batch) correspond au nombre de flux, ici 409 par exemple
        # → Donc la forme en entrée est : (nb_flux, longueur_max_chemin, path_state_dim)
        # Cette readout mappe chaque vecteur de dimension 32 à un scalaire via des couches denses
        # Elle renvoie donc un tenseur de forme (nb_flux, longueur_max_chemin, 1)
        self.readout_path = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(int(self.link_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.path_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")

    @tf.function
    def call(self, inputs):
        print("normal model")
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model']
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


        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

        pkt_size = traffic / packets

        # Initialize the initial hidden state for links
        path_state = self.path_embedding(tf.concat(
            [(traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
             (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
             tf.one_hot(model, self.max_num_models),
             (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
             (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
             (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
             (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
             (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
             (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
             (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
             (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1]], axis=1))

        # Initialize the initial hidden state for paths
        link_state = self.link_embedding(tf.concat([load, policy], axis=1)) #shape = (71,2)

        # Initialize the initial hidden state for paths
        queue_state = self.queue_embedding(
            tf.concat([(queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
                       priority, weight], axis=1)) #shape = (111,3)

        # Iterate t times doing the message passing
        for it in range(self.iterations):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            # Récupération des états des files d'attente (queues) pour chaque flux
            # → queue_gather aura la forme (nb_flux, longueur_du_chemin, queue_state_dim)
            # Chaque flux traverse un certain nombre de queues, dépendant de sa longueur (nombre de sauts dans le graphe)
            # Exemple : si le flux n°1 traverse 3 queues, alors queue_gather[0] contient 3 vecteurs (un par queue traversée)
            # Cela représente donc la description du flux du point de vue des queues
            queue_gather = tf.gather(queue_state, queue_to_path)
            # Même logique pour les liens :
            # Récupération des états des liens pour chaque flux, selon le chemin emprunté
            # → link_gather aura également la forme (nb_flux, longueur_du_chemin, link_state_dim)
            # Cette opération fournit une description du flux du point de vue des liens qu'il traverse
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")

            # À chaque itération de message passing, on instancie un RNN basé sur self.path_update (ici une cellule GRU ou LSTM)
            # Cela permet de traiter séquentiellement les couples (lien, queue) que chaque flux traverse

            # Le GRU (ou LSTM) utilisé ici permet d’intégrer une mémoire sur l’ensemble du chemin :
            # Il capture les dépendances temporelles entre les étapes successives du flux.
            # Par exemple : si un lien était vide à l’étape précédente, il y a de fortes chances que le suivant le soit aussi pour un flux donné.

            # Contrairement à un simple RNN sans mémoire longue, ici le GRU maintient une trace de tout l’historique des étapes traversées
            # Cela permet au modèle de contextualiser chaque étape avec tout ce qui a été vu avant (et non uniquement l’état précédent).
            #Il est instancié qu'une fois, et c'est lors de l'initialisation

            # En parallèle, le RNN en lui-même permet aussi de lier l’état du flux à l’itération précédente à l’ensemble des états actuels
            # (queues et liens traversés dans cette itération de message passing).
            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)
            previous_path_state = path_state # path_state.shape = (nbr_flux, dim_flux (=self.path_state_dim))
            # On concatène les embeddings des queues et des liens pour chaque étape du chemin
            # → Forme du tenseur : (nb_flux, longueur_du_chemin, dim_queue + dim_link)
            # La deuxième dimension est variable (None), car chaque flux peut avoir un chemin de longueur différente

            # On peut concaténer directement car chaque étape du chemin correspond à un couple (queue, lien)

            # Cette séquence est ensuite traitée par le GRU via path_update_rnn
            # Le GRU parcourt chaque chemin, étape par étape, pour apprendre la dynamique du flux dans le réseau
            # → Cela permet de capturer le fait, par exemple, qu’un lien congestionné à t−1 risque de l’être encore à t

            # Résultat :
            # - path_state_sequence : la séquence complète d’états pour chaque étape (utilisée ensuite pour la mise à jour des queues)
            # - path_state : le dernier état (utilisé comme mémoire pour l’itération suivante du message passing)
            path_state_sequence, path_state = path_update_rnn(tf.concat([queue_gather, link_gather], axis=2),
                                                              initial_state=path_state) #Mémoire n-1 sur le RNN, mais avec le GRU on garde un historique RELEVANT

            # La variable `path_state_sequence` contient la séquence des états du flux après passage dans le GRU
            # Sa forme est : (nb_flux, longueur_du_chemin, path_state_dim), avec path_state_dim = 32
            # → La longueur du chemin varie selon les flux, mais est bornée par un padding à 5 (longueur max fixée)

            # On ajoute manuellement l’état initial du flux (`previous_path_state`) en début de séquence,
            # en l’expandant sur un axe temporel : shape (nb_flux, 1, path_state_dim)
            # → On le concatène à `path_state_sequence`, ce qui donne une forme finale de (nb_flux, 6, path_state_dim)

            # Cette concaténation revient à considérer l’état initial comme une "étape fictive"
            # représentant le contexte global du flux avant de traverser les couples (queue, lien)

            # Résultat : on obtient, pour chaque flux, une séquence d’états de longueur 6 (1 initial + 5 étapes max),
            # où chaque état encode une étape (queue, link) du flux avec une représentation vectorielle de taille 32
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence],
                axis=1
            )

            ###################
            #  PATH TO QUEUE  #
            ###################
            # Pour chaque queue, on récupère les états des flux qui la traversent
            # Attention : on ne récupère pas l’état complet du flux,
            # mais uniquement l’état associé à l’étape du flux où cette queue est impliquée
            # Cela se fait grâce à `path_to_queue`, qui contient les indices (flux_id, position_dans_le_flux)

            # Résultat : path_gather est un tenseur de forme (nb_queues, nb_flux_max_par_queue, path_state_dim)
            # Il contient pour chaque queue les représentations locales des flux qui la traversent
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)

            # On agrège les contributions de tous les flux vers chaque queue par une somme
            # → Chaque queue reçoit un résumé des états des flux qui la traversent
            path_sum = tf.math.reduce_sum(path_gather, axis=1)

            # On met ensuite à jour l’état de la queue à partir de cette agrégation
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            # Chaque lien est influencé uniquement par l’ensemble des queues qui lui sont associées (via queue_to_link)
            # On récupère donc, pour chaque lien, les états des queues qui l’alimentent
            # → queue_gather a pour forme : (nb_liens, nb_queues_par_lien, queue_state_dim)

            queue_gather = tf.gather(queue_state, queue_to_link)

            # On applique ensuite un RNN (ici un GRU ou LSTM) sur les séquences de queues associées à chaque lien
            # Cela permet d’agréger dynamiquement l'information issue des queues pour produire un nouvel état du lien
            # `return_sequences=False` → on ne garde que l’état final du RNN (résumé global des queues)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        #################### #################### #################### #################### ####################
        #################### Partie centrale : prédiction du délai de bout-en-bout par flux ####################
        #################### #################### #################### #################### ####################

        # C’est ici que se joue la singularité du modèle (delay, loss, jitter)
        # La prédiction se fait au niveau du flux (readout flow-level) en combinant la congestion et la transmission.

        # On commence par récupérer les capacités des liens traversés par chaque flux
        # → Grâce à link_to_path, on mappe chaque chemin à sa séquence de liens
        capacity_gather = tf.gather(capacity, link_to_path)

        # On extrait les états latents du flux à chaque étape, en excluant le premier timestep (état initial ajouté manuellement)
        # → input_tensor a pour shape : (nb_flux, longueur_max = 5, path_state_dim = 32)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        # On applique la fonction de readout pour prédire une "queue occupancy estimée" à chaque étape
        # → occupancy_gather est la file d’attente effective à chaque saut du chemin
        occupancy_gather = self.readout_path(input_tensor)

        # On transforme cette séquence dense en RaggedTensor pour prendre en compte les flux de longueur variable
        length = tf.ensure_shape(length, [None])  # taille réelle de chaque flux
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        ### Intuition derrière tout cela :

        # occupancy_gather : estimation du nombre de bits/paquets en attente dans chaque queue traversée
        # capacity_gather : capacité de chaque lien traversé (en bits/s ou packets/s)

        # En divisant les deux, on obtient une estimation du délai d’attente dans chaque file (file occupancy / débit)
        # Puis on somme sur tous les liens → délai de queue cumulé
        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)

        # En parallèle, on ajoute un délai de transmission, qui dépend uniquement :
        # - de la taille moyenne des paquets
        # - du débit des liens traversés (1 / capacité)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        # Prédiction finale : délai total = délai de queue + délai de transmission
        return queue_delay + trans_delay
