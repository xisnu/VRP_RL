import tensorflow as tf
from VRP_Env import *
from VRP_Data import *


# query N,dim
# ref N,T,encoder_dim
# Env instance
class VRPAttention(tf.keras.Model):

    def __init__(self, dim):
        super(VRPAttention, self).__init__()
        self.emb_d = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='emb_demand_vrp_attention')
        self.emb_ld = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='emb_load_demand_vrp_attention')
        self.project_d = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='project_demand_vrp_attention')
        self.project_ld = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='project__load_demand_vrp_attention')
        self.project_ref = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='project_ref_vrp_attention')
        self.project_query = tf.keras.layers.Dense(dim, activation='relu', name='project_ref_vrp_attention')
        self.V = tf.keras.layers.Dense(1, name='V_logit_vrp_attention', kernel_initializer='uniform')

    def call(self, query, ref, env, training=None, mask=None):
        """
                This function gets a query tensor and ref rensor and returns the logit op.
                Args:
                    query: is the hidden state of the decoder at the current
                        time step. [batch_size x dim]
                    ref: the set of hidden states from the encoder.
                        [batch_size x max_time x dim]

                Returns:
                    e: convolved ref with shape [batch_size x max_time x dim]
                    logits: [batch_size x max_time]
                """
        demand = env.demand  # demand N,T
        T = tf.shape(demand)[1]
        load = env.load  # load = N,
        # demand N,T
        demand_exp = tf.expand_dims(demand, 2)  # N,T,1
        emb_d = self.emb_d(demand_exp)  # N,T,dim
        d = self.project_d(emb_d)  # N,T,dim
        # load = N,
        load = tf.expand_dims(load, 1)  # N,1
        tiled_load = tf.tile(load, [1, T - 1])  # N,T-1
        load_demand = tiled_load - demand[:, :T - 1]  # N,T-1
        temp = tf.zeros([load_demand.shape[0], 1])
        load_demand = tf.concat([load_demand, temp], -1)
        load_demand = tf.expand_dims(load_demand, 2)  # N,T,1
        emb_ld = self.emb_ld(load_demand)  # N,T,dim
        ld = self.project_ld(emb_ld)  # N,T,dim
        # start computing attention
        # ref N,T,encoder_dim
        e = self.project_ref(ref)  # N,T,dim
        # query N,dim (decoder state)
        q = self.project_query(query)  # N,dim
        q_exp = tf.expand_dims(q, 1)  # N,1,dim
        q_exp = tf.tile(q_exp, [1, T, 1])  # N,T,dim
        # Now everything has same dimension N,T,dim
        logit = d + ld + e + q_exp  # N,T,dim
        # Project through V
        logit = self.V(logit)  # N,T,1
        logit = tf.squeeze(logit, 2)  # N,T

        return e, logit


# query N,dim
# ref N,T,encoder_dim
# Env instance
class VRPCritic(tf.keras.Model):
    def __init__(self, dim):
        super(VRPCritic, self).__init__()
        self.emb_d = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='emb_demand_vrp_critic')
        self.emb_ld = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='emb_load_demand_vrp_critic')
        self.project_d = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='project_demand_vrp_critic')
        self.project_ld = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='project__load_demand_vrp_critic')
        self.project_ref = tf.keras.layers.Conv1D(dim, 1, activation='relu', name='project_ref_vrp_critic')
        self.project_query = tf.keras.layers.Dense(dim, activation='relu', name='project_ref_vrp_critic')
        self.V = tf.keras.layers.Dense(1, name='V_logit_vrp_critic', kernel_initializer='uniform')
        self.linear_project = tf.keras.layers.Dense(1, activation='relu', name='V_linproj_vrp_critic',
                                                    kernel_initializer='uniform')

    def call(self, query, ref, env, training=None, mask=None):
        demand = env.demand  # demand N,T
        T = tf.shape(demand)[1]
        emb_d = self.emb_d(tf.expand_dims(demand, -1))  # N,T,dim
        d = self.project_d(emb_d)  # N,T,dim
        e = self.project_ref(ref)  # N,T,dim
        q = self.project_query(query)  # N,dim
        q_exp = tf.expand_dims(q, 1)  # N,1,dim
        q_exp = tf.tile(q_exp, [1, T, 1])  # N,T,dim
        u = self.V(q_exp + e + d)  # N,T,1
        logit = tf.squeeze(u, 2)  # N,T
        logit = self.linear_project(logit)
        return e, logit


# inputs N,T,3 (coordinates + demand)
class VRPEncoder(tf.keras.Model):
    def __init__(self, emb_dim):
        super(VRPEncoder, self).__init__()
        self.project_emb = tf.keras.layers.Conv1D(emb_dim, 1, padding='same', activation='relu',
                                                  name='embedding_vrp_encoder')

    def call(self, inputs, training=None, mask=None):
        emb = self.project_emb(inputs)
        return emb


class VRPRL:
    def __init__(self, enc_emb_dim):
        self.config = read_config("VRP_Config")
        self.dim = enc_emb_dim
        self.env = Env(self.config)
        self.vrp_data_gen = DataGenerator(self.config)
        self.encoder = VRPEncoder(enc_emb_dim)
        self.decoder = VRPAttention(enc_emb_dim)
        self.critic = VRPCritic(enc_emb_dim)
        self.optimizer = tf.keras.optimizers.Adam(0.0001)

    def train(self, max_len, selection='stochastic', epochs=50):
        for e in range(epochs):
            vrp_train = self.vrp_data_gen.get_train_next()  # N,T,3
            batch_size = vrp_train.shape[0]
            all_actions = []
            all_log_probs = []
            with tf.GradientTape() as a_tape, tf.GradientTape() as c_tape:
                init_state = self.env.reset(vrp_train)
                # Now we get the encoder output which is embedding on coordinates + demand
                encoder_emb = self.encoder(vrp_train)  # N,T,enc_emb_dim
                # Now decoder loop will start , we need an initial decoder input
                decoder_input = encoder_emb[:, -1]  # N,emc_emb_dim [last data from embedding (depot) where demand = 0]
                batch_sequence = tf.expand_dims(tf.range(batch_size), 1)  # N,1
                for t in range(max_len):
                    e_t, logit_t = self.decoder(decoder_input, encoder_emb, self.env)
                    log_prob_t = tf.nn.log_softmax(logit_t)
                    prob_t = tf.exp(log_prob_t)
                    if selection == 'stochastic':
                        idx = tf.random.categorical(logit_t, 1)  # N,1
                        idx_exp = tf.cast(idx, tf.int32)  # N,1
                    else:
                        idx = tf.argmax(prob_t, axis=-1)  # N,
                        idx_exp = tf.cast(tf.expand_dims(idx, 1), tf.int32)  # N,1
                    gather_indices = tf.concat([batch_sequence, idx_exp], axis=-1)  # N,2
                    decoder_input = tf.gather_nd(encoder_emb, gather_indices)
                    new_state = self.env.step(idx_exp, vrp_train)
                    action = tf.gather_nd(self.env.input_pnt, gather_indices)
                    all_log_probs.append(tf.gather_nd(prob_t, gather_indices))
                    all_actions.append(action)
                R = reward_func(all_actions)
                # Now run critic
                query = tf.zeros([batch_size, self.dim], tf.float32)
                _, v = self.critic(query, encoder_emb, self.env)
                v_nograd = tf.stop_gradient(v)
                c_loss = tf.keras.losses.mean_squared_error(R, v)
                a_loss = tf.reduce_mean(tf.multiply((R - v_nograd), tf.add_n(all_log_probs)), 0)
                a_vars = self.decoder.trainable_variables
                c_vars = self.critic.trainable_variables
                a_grads = a_tape.gradient(a_loss, a_vars)
                c_grads = c_tape.gradient(c_loss, c_vars)
                mean_reward = np.mean(R)
            self.optimizer.apply_gradients(zip(a_grads, a_vars))
            self.optimizer.apply_gradients(zip(c_grads, c_vars))
            print("Epoch %d/%d Average Reward %f" % (e,epochs,mean_reward))


network = VRPRL(4)
network.train(8,epochs=15000)
