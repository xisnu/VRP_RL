import tensorflow as tf
import collections

tf.random.set_seed(123)

class State(collections.namedtuple("State",
                                        ("load",
                                         "demand",
                                         'd_sat',
                                         "mask"))):
    pass

class Env(object):
    def __init__(self,
                 args):
        '''
        This is the environment for VRP.
        Inputs:
            args: the parameter dictionary. It should include:
                args['n_nodes']: number of nodes in VRP
                args['n_custs']: number of customers in VRP
                args['input_dim']: dimension of the problem which is 2
        '''
        self.capacity = args['capacity']
        self.n_nodes = args['n_nodes']
        self.n_cust = args['n_cust']
        self.input_dim = args['input_dim']
        # self.input_data = tf.placeholder(tf.float32,   shape=[None, self.n_nodes, self.input_dim])


    def reset(self,input_data, beam_width=1):
        '''
        Resets the environment. This environment might be used with different decoders.
        In case of using with beam-search decoder, we need to have to increase
        the rows of the mask by a factor of beam_width.
        '''
        self.input_pnt = input_data[:, :, :2]  # not taking demand value from last axis
        self.demand = input_data[:, :, -1]  # this is only demand value
        self.batch_size = tf.shape(self.input_pnt)[0]
        # dimensions
        self.beam_width = beam_width
        self.batch_beam = self.batch_size * beam_width


        # modify the self.input_pnt and self.demand for beam search decoder
        #         self.input_pnt = tf.tile(self.input_pnt, [self.beam_width,1,1])

        # demand: [batch_size * beam_width, max_time]
        # demand[i] = demand[i+batchsize]
        self.demand = tf.tile(self.demand, [self.beam_width, 1])#N.Beam , max_time

        # load: [batch_size * beam_width]
        self.load = tf.ones([self.batch_beam]) * self.capacity#N.Beam,capacity

        # create mask
        self.mask = tf.zeros([self.batch_size * beam_width, self.n_nodes],
                             dtype=tf.float32)

        # update mask -- mask if customer demand is 0 and depot
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.ones([self.batch_beam, 1])], 1)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=tf.zeros([self.batch_beam, self.n_nodes]),
                      mask=self.mask)

        return state

    def step(self,
             idx,input_data,
             beam_parent=None):
        '''
        runs one step of the environment and updates demands, loads and masks
        '''

        # if the environment is used in beam search decoder
        if beam_parent is not None:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = tf.expand_dims(tf.tile(tf.cast(tf.range(self.batch_size), tf.int64),
                                                  [self.beam_width]), 1)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx = batchBeamSeq + tf.cast(self.batch_size, tf.int64) * beam_parent
            # demand:[batch_size*beam_width x sourceL]
            self.demand = tf.cast(tf.gather_nd(self.demand, batchedBeamIdx),tf.float32)
            # load:[batch_size*beam_width]
            self.load = tf.gather_nd(self.load, batchedBeamIdx)
            # MASK:[batch_size*beam_width x sourceL]
            self.mask = tf.gather_nd(self.mask, batchedBeamIdx)

        # self.demand = input_data[:, :, -1]

        BatchSequence = tf.expand_dims(tf.cast(tf.range(self.batch_beam), tf.int32), 1)
        batched_idx = tf.concat([BatchSequence, idx], 1)

        # how much the demand is satisfied
        collect = tf.gather_nd(self.demand, batched_idx)
        d_sat = tf.minimum(collect, self.load)

        # update the demand
        d_scatter = tf.scatter_nd(batched_idx, d_sat, tf.cast(tf.shape(self.demand), tf.int32))
        self.demand = tf.subtract(self.demand, d_scatter)

        # update load
        self.load -= d_sat

        # refill the truck -- idx: [10,9,10] -> load_flag: [1 0 1]
        load_flag = tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32), 1)
        self.load = tf.multiply(self.load, 1 - load_flag) + load_flag * self.capacity

        # mask for customers with zero demand
        self.mask = tf.concat([tf.cast(tf.equal(self.demand, 0), tf.float32)[:, :-1],
                               tf.zeros([self.batch_beam, 1])], 1)

        # mask if load= 0
        # mask if in depot and there is still a demand

        self.mask += tf.concat([tf.tile(tf.expand_dims(tf.cast(tf.equal(self.load, 0),
                                                               tf.float32), 1), [1, self.n_cust]),
                                tf.expand_dims(
                                    tf.multiply(tf.cast(tf.greater(tf.reduce_sum(self.demand, 1), 0), tf.float32),
                                                tf.squeeze(tf.cast(tf.equal(idx, self.n_cust), tf.float32))), 1)], 1)

        state = State(load=self.load,
                      demand=self.demand,
                      d_sat=d_sat,
                      mask=self.mask)

        return state

def reward_func(sample_solution):
    """The reward for the VRP task is defined as the
    negative value of the route length
    Args:
        sample_solution : a list tensor of size decode_len of shape [batch_size x input_dim]
        demands satisfied: a list tensor of size decode_len of shape [batch_size]
    Returns:
        rewards: tensor of size [batch_size]
    Example:
        sample_solution = [[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]
        sourceL = 3
        batch_size = 2
        input_dim = 2
        sample_solution_tilted[ [[5,5]
                                                    #  [6,6]]
                                                    # [[1,1]
                                                    #  [2,2]]
                                                    # [[3,3]
                                                    #  [4,4]] ]
    """
    # make init_solution of shape [sourceL x batch_size x input_dim]


    # make sample_solution of shape [sourceL x batch_size x input_dim]
    sample_solution = tf.stack(sample_solution,0)

    sample_solution_tilted = tf.concat((tf.expand_dims(sample_solution[-1],0),
         sample_solution[:-1]),0)
    # get the reward based on the route lengths


    route_lens_decoded = tf.reduce_sum(tf.pow(tf.reduce_sum(tf.pow(\
        (sample_solution_tilted - sample_solution) ,2), 2) , .5), 0)
    return route_lens_decoded