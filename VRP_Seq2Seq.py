
from VRP_Env import *
from VRP_Data import *
from VRP_Models import *

class VRPRL:
    def __init__(self, enc_emb_dim):
        self.config = read_config("VRP_Config")
        self.dim = enc_emb_dim

        self.env = Env(self.config)
        self.vrp_data_gen = DataGenerator(self.config)
        self.encoder = VRPEncoder(enc_emb_dim)
        self.decoder = VRPDecoder(enc_emb_dim)
        self.critic = VRPCritic(enc_emb_dim)
        self.optimizer = tf.keras.optimizers.Adam(0.0001)
        self.checkpoint = tf.train.Checkpoint(encoder = self.encoder, decoder= self.decoder,critic=self.critic)
        self.weight_manager = tf.train.CheckpointManager(self.checkpoint,"Weights/VRP",max_to_keep=2)

    def train(self, max_len, selection='stochastic', epochs=50):
        latest_checkpoint = tf.train.latest_checkpoint("Weights/VRP")
        self.checkpoint.restore(latest_checkpoint)
        print("Weights restored from ",latest_checkpoint)
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
                # and an initial decoder state
                decoder_state = tf.zeros([batch_size,self.dim]) # as we are using GRU
                batch_sequence = tf.expand_dims(tf.range(batch_size), 1)  # N,1
                for t in range(max_len):
                    logit_t, decoder_state = self.decoder(decoder_input,decoder_state,encoder_emb, self.env)
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
            self.weight_manager.save()


network = VRPRL(4)
network.train(8,epochs=2000,selection='greedy')
