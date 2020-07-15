import tensorflow as tf


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

# VRP Decoder with Attention
class VRPDecoder(tf.keras.Model):
    def __init__(self, dim):
        super(VRPDecoder, self).__init__()
        self.attention = VRPAttention(dim)
        self.gru = tf.keras.layers.GRU(dim,return_state=True,return_sequences=True)
        self.INFINITE = 100000

    # implementation of one step of decoder
    def call(self, decoder_inp,decoder_state,context,env, training=None, mask=None):
        decoder_inp = tf.expand_dims(decoder_inp,1) # include a time dimension
        gru_out, gru_state = self.gru(decoder_inp,initial_state=decoder_state)
        # gru_out will not be used, gru state will be used in attention
        e, logit = self.attention(gru_state, context, env) # unmasked
        logit -= self.INFINITE * env.mask # masked
        return logit, gru_state
