import tensorflow as tf


class RouteNet_Fermi(tf.keras.Model):
    def __init__(self):
        super(RouteNet_Fermi, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_num_models = 7

        self.num_policies = 4
        self.max_num_queues = 3

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

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
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

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

        self.readout_path = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(int(self.link_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.path_state_dim / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")


    def call(self, inputs):
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
            # Etat des queues pour chaque flux, donc de shape : (nbr_flux,1,queue_dim)
            # Donc chaque element est de longueur de la longueur du flux
            # Par exemple, le flux n°1, est de longueur 3, donc le premier element possède 3 etats de queues (les 3 par lequel le flux passe)
            #description du flux en terme de queues
            queue_gather = tf.gather(queue_state, queue_to_path)
            #Même chose que queue_gather
            #Pour chaque flux, état de liens par lequel le flux passe
            # description du flux en terme de links
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")

            # On redéfinit à chaque itération le path update en gardant self.path_update
            #En gros, le rnn à une mémoire jusqu'à N-1, alors que le GRU à une mémoire depuis le tout début (non re instancié contrairement au RNN)
            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)
            previous_path_state = path_state # path_state.shape = (nbr_flux, dim_flux (=self.path_state_dim))
            #tf.concat([queue_gather, link_gather], axis=2).shape = (nbr_flux, None, dim_queue+dim_link) -> on combine les vecteurs, None car dépend de la longueur
            #flux, le gru + rnn normalise cela en prenant la longueur max = 5
            #on peut le faire car autant de lien que de queues par flux
            #Un chemin congestionné à t-1 a des chances d’être encore congestionné à t
            path_state_sequence, path_state = path_update_rnn(tf.concat([queue_gather, link_gather], axis=2),
                                                              initial_state=path_state) #Mémoire n-1 sur le RNN, mais avec le GRU on garde un historique RELEVANT
            # La dimension de path_state_sequence dépend du nombre de time steps dans le RNN, ie de la longueur du flux
            #shape = (nbr_flux, 5, self.path_state_dim (=GRU dim)), 5 étant la longueur maximale du flux !
            #On rajoute l'état initial pour avoir une shape (nbr_flux,6,self.path_state_dim (=GRU dim))
            # description des flux, qui prend en compte à la fois la queue et le link, pour chaque (queue,link) par lequel le flux passe, ie maximum 5
            path_state_sequence = tf.concat([tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1)
            #un peu comme si, l'état initial était pris en compte comme un lien précédent tous les autres liens !
            #on a donc plus une longueur de 5 mais de 6 !
            #On a bien tous les états des étapes qui sont de dimensions 32
            ###################
            #  PATH TO QUEUE  #
            ###################
            #Pour chaque queue, on récupere les états des flow associés à l'indice adéquat
            #Attention on ne récupère pas le flow en entier ! Seulement la partie du flow qui correspond à la queue associée !
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue) #shape (nbr_queues, nbr_max_flux_par_queue , self.state_queue)
            tf.print(path_gather)

            path_sum = tf.math.reduce_sum(path_gather, axis=1) #On somme la contributions de l'ensemble des flows à la queue donnée
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            #Le lien dépend seulement de l'ensemble des queues dont il dépend. On prend l'esimation ci dessus de queue_state
            queue_gather = tf.gather(queue_state, queue_to_link) #On obtient, pour chaque lien, l'ensemble des états des queues qui le compose

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)


        capacity_gather = tf.gather(capacity, link_to_path)

        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather,
                                         axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        return queue_delay + trans_delay
