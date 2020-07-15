import numpy as np
import tensorflow as tf
import os
import warnings
import collections
np.random.seed(123)

def read_config(config_file):
    config = {}
    f =open(config_file)
    line = f.readline()
    while line:
        info = line.strip("\n").split(":")
        param_name = info[0]
        param_value = int(info[1])
        config[param_name] = param_value
        line = f.readline()
    f.close()
    return config

def create_VRP_dataset(
        n_problems,
        n_cust,
        data_dir,
        seed=None,
        data_type='train'):
    '''
    This function creates VRP instances and saves them on disk. If a file is already available,
    it will load the file.
    Input:
        n_problems: number of problems to generate.
        n_cust: number of customers in the problem.
        data_dir: the directory to save or load the file.
        seed: random seed for generating the data.
        data_type: the purpose for generating the data. It can be 'train', 'val', or any string.
    output:
        data: a numpy array with shape [n_problems x (n_cust+1) x 3]
        in the last dimension, we have x,y,demand for customers. The last node is for depot and
        it has demand 0.
     '''

    # set random number generator
    n_nodes = n_cust + 1
    if seed == None:
        rnd = np.random
    else:
        rnd = np.random.RandomState(seed)

    # build task name and datafiles
    task_name = 'vrp-size-{}-len-{}-{}.txt'.format(n_problems, n_nodes, data_type)
    fname = os.path.join(data_dir, task_name)

    # cteate/load data
    if os.path.exists(fname):
        print('Loading dataset for {}...'.format(task_name))
        data = np.loadtxt(fname, delimiter=' ')
        data = data.reshape(-1, n_nodes, 3)
    else:
        print('Creating dataset for {}...'.format(task_name))
        # Generate a training set of size n_problems
        x = rnd.uniform(0, 1, size=(n_problems, n_nodes, 2))
        d = rnd.randint(1, 10, [n_problems, n_nodes, 1])
        d[:, -1] = 0  # demand of depot
        data = np.concatenate([x, d], 2)
        np.savetxt(fname, data.reshape(-1, n_nodes * 3))

    return data

class DataGenerator(object):
    def __init__(self,
                 args):

        '''
        This class generates VRP problems for training and test
        Inputs:
            args: the parameter dictionary. It should include:
                args['random_seed']: random seed
                args['test_size']: number of problems to test
                args['n_nodes']: number of nodes
                args['n_cust']: number of customers
                args['batch_size']: batchsize for training
        '''
        self.args = args
        self.rnd = np.random.RandomState(seed=args['random_seed'])
        print('Created train iterator.')

        # create test data
        self.n_problems = args['test_size']
        self.test_data = create_VRP_dataset(self.n_problems, args['n_cust'], 'Data',
                                            seed=args['random_seed'] + 1, data_type='test')

        self.reset()

    def reset(self):
        self.count = 0

    def get_train_next(self):
        '''
        Get next batch of problems for training
        Retuens:
            input_data: data with shape [batch_size x max_time x 3]
        '''

        input_pnt = self.rnd.uniform(0, 1,
                                     size=(self.args['batch_size'], self.args['n_nodes'], 2))

        demand = self.rnd.randint(1, 10, [self.args['batch_size'], self.args['n_nodes']])
        demand[:, -1] = 0  # demand of depot

        input_data = np.concatenate([input_pnt, np.expand_dims(demand, 2)], 2).astype('float32')

        return input_data

    def get_test_next(self):
        '''
        Get next batch of problems for testing
        '''
        if self.count < self.args['test_size']:
            input_pnt = self.test_data[self.count:self.count + 1]
            self.count += 1
        else:
            warnings.warn("The test iterator reset.")
            self.count = 0
            input_pnt = self.test_data[self.count:self.count + 1]
            self.count += 1

        return input_pnt

    def get_test_all(self):
        '''
        Get all test problems
        '''
        return self.test_data

