from VRP_Data import *
from VRP_Env import *

config = read_config("VRP_Config")
vrp_data_gen = DataGenerator(config)
vrp_train = vrp_data_gen.get_train_next()
print(vrp_train[0,:3])
# print(vrp_train[0,0:3])

# vrap_train_wo_demand = vrp_train[:,:,:2]
vrp_10_env = Env(config,vrp_train)
init_state = vrp_10_env.reset()
print(init_state)


print("After One Step")
idx = np.expand_dims([3,1],-1)
state = vrp_10_env.step(idx)
print(state)


print("After Two Steps")
idx = np.expand_dims([2,2],-1)
state = vrp_10_env.step(idx)
print(state)

# sample_solution = np.asarray([[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]]).astype('float32') #3,N,2 -- N=2
sample_solution = np.asarray([[[0,0]],[[2,2]],[[3,4]]]).astype('float32')# 3,N,2 -- N=1
route_len = reward_func(sample_solution)
print(route_len)