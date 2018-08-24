"""
distributioncenter environments have the following traits in common:
- A demand and supply modeling , where the demand on the distribution center for different products is modeled , supply is always greater by a constant margin
- A setup of the distribution center which can be set up by sampling operations according to a probability distribution whenever a demand can't be
served by the current setup e.g. if there is too much demand on a product and the corresponding storage can't carry this amount of product
- Agents effectively only take actions if the demand can't be served and control the setup of the distribution center
- observation consists of a demand/supply/setup combination of the distribution center
- there might be illegal states like a state minimum isle size upon which a new action is sampled instead


Actions consist of 3 sub-actions:
    - ID of the storage/buffer that is resized
    - magnitude of change in size
    - Whether to write to the output tape
    - Which character to write (ignored if the above sub-action is 0)
An episode ends when:
    - The agent can serve the demand for an entire episode .
    - The agent can't serve the demand anymore cause the supply is empty.
    - The agent runs out the time limit. (Which is fairly conservative.)
Reward schedule:
# defined as a function of throughput speed
    # high reward for
In the beginning, demands will be small for the overall setup, after an environement has consistently served the demand
over some episodes, the environment will increase the demand on the distribution center. Typically many times levelling up the demand will be necessary
until the reward threshold is reached.
Reward : we want to minimize the lift time and transport time. For that our objective function is defined as the negative transport and lift times for a certain demand and maximizing
this quantity will minimize the lift and transport times.
"""




import numpy as np
import gym
from importlib import reload
from gym import error, spaces, utils
from gym.utils import seeding
import os, subprocess, time, signal, logging
#from gym_dc.envs import DistributionCenter
#from gym_dc.envs import observation_space
#from gym_dc.envs import action_space
from gym_dc.envs import demandmodel

reload(demandmodel)

logger = logging.getLogger(__name__)




class DCEnv(gym.Env):

    def __init__(self):
        # Static variables defining the environment
        self.config = {'action' : ['numlifts', 'numtransports', 'sizelifts', 'sizetransports', 'queuelift', 'queuetransport', 'demand', 'supply'],
        'state': ['lift_resize', 'add_rem_lift', 'add_rem_transport', 'transport_resize']}

        # RL_ agent boolean indicates if action is to be sampled in this state or not
        self.RL_boolean = False
        #self.reward = None
        self.policy = []
        self.max_area_storage = 4000
        # self.max_len_buffer = 60
        # self.max_width_buffer = 60
        # self.max_elements_buffer = 500
        # self.max_len_DC = 7000
        # self.max_width_DC = 5000
        self.min_width_isle = None
        self.num_isle = 2
        self.isle_margin = 2
        # area in squarecm
        self.area_DC = 10000
        self.filename = 'demand.csv'
        self.demand = []
        self.supply = []
        self.total_supply = sum(self.supply)
        self.total_demand = None
        self.max_elements_storages = [0] * len(self.demand)
        self.total_elements_storage = sum(self.max_elements_storages)
        # default storage size 500 squarecm
        self.size_storages = [500] * len(self.supply)
        # machines that carry the supply from A to B
        self.num_lifts = 10
        self.num_transports = 10
        self.max_numlifts = 30
        self.max_numtransports = 30
        self.lift_size = 20
        self.product_size = 10
        self.max_size_transport = self.product_size * 100
        self.max_size_lift = self.product_size * 10
        # one transport can carry
        self.transport_size = 100
        self.num_products_transport = int(self.transport_size/self.product_size)
        # queue if too many transports want to move towards exit
        self.transport_queue = [0] * self.num_transports
        # queue if too many products need to be retrieved with same lift
        self.lift_queue = [0] * self.num_lifts
        # for each lift we have a constant time to retrieve a certain product from storage
        self.time_lift_retrieve = [1] * self.num_lifts
        # for each transport we have a time to transport to exit -> if there is a queue at the exit a constant multiplied by queue length is added to each of these times
        self.time_transport_exit = [2] * self.num_lifts
        # levels
        self.storage_level = None
        #self.buffer_level = None
        self.lift_occupancy = None
        self.transport_occupancy = None
        # self state vector for individual parts
        #self.state_buffers = np.array([])
        self.state_storages = []
        self.state_lifts = []
        self.state_transports = []
        # complete state vector describing DC
        self.StateVector = None

        # additional transport/lift constant time is dependent on the number of transports and lifts
        # the more transports/lifts the longer the time as they might block each other
        self.lift_constant = self.num_lifts * 0.1
        self.transport_constant = self.num_transports * 0.1


        # # observation space of env
        # self.observation_space = observation_space.observation_space(self)
        #  # action space of env
        #self.action_space = action_space.action_space(self)
        # Discrete spaces

        # Sequence in action space:
        #TODO: once this works expand action space
        # 1. add/remove lift
        # 2. add/remove transport
        # lowest are the lowest accepted values
        # highest are the highest accepted values will be a 4 x 4 matrix of actions
        # Multidiscrete lower and upper bounds possible lower leave at least one lift/transport
        # resize and leave at least product size space on transport and lift
        #self.action_space = spaces.Tuple((spaces.MultiDiscrete(-(self.num_lifts-1), (self.max_numlifts - self.num_lifts))))
        # this is the continuous feature space that gives a 4 dimensional input for actions
        lowarr = np.array([-(self.lift_size-self.product_size), -(self.num_lifts-1),-(self.num_transports-1), -(self.transport_size-self.product_size)])
        higharr = np.array([(self.max_size_lift - self.lift_size), (self.max_numlifts - self.num_lifts), (self.max_numtransports - self.num_transports),(self.max_numtransports - self.num_transports)])
        self.action_space = spaces.Box(low=lowarr, high=higharr, dtype='int')
        # , spaces.MultiDiscrete(-(self.num_transports-1), (self.max_numtransports - self.num_transports),1), \
        # spaces.MultiDiscrete(-(self.transport_size-self.product_size), (self.max_numtransports - self.num_transports),1)))

        # state of the environment, numobservations = dimensionality of state vector
        # state vector contains: [numlifts, numtransports, sizelifts, sizetransports, queuelift, queuetransport, demand, supply]
        self.num_observations = 8
        lowarr = np.array([0]*self.num_observations).astype('float64')
        higharr = np.array([10000]*self.num_observations) # do not!! use np.inf
        self.observation_space = spaces.Box(low=lowarr, high=higharr, dtype='int')
        # for RL variables
        self.RL_boolean = True
        self.time = 0
        self.timelimit = 1000
        self.episode_total_reward = 0
        self.reward = 0
        # discretize state space for q-table, instead we use Q-learning where we use function approximation such that Q values are estimated
        # taking dot product between weights and feature vector (state, action) pair
        self.Q_table = {}

        #self.transport_times = None
        #self.lift_times = None









##############################################
#
# Compute reward : defined as minimum time to get all products combined to the exit
# This time is affected by: where the product is stored, how many lifts can access the type of product, and how many transports can transport product to exit
# there is also a queue if too many transports try to travel the same route
#####################################





    ##############################################
    #
    # RL Agent functions
    #
    #####################################


    def step(self, action):
        ''' next state in MDP if action has to be taken from RL agent simulate a step and compute reward'''
        # TODO move down to if RL_boolean
        self.last_action = action
        done = False
        reward = 0.0
        self.time += 1

        if self.time == self.timelimit:
            done = True
            info = None
            statevector = None
            return statevector, reward, done, info
        else:
            # sample new state
            state = self.observation_space.sample()
            #action = self.action_space.sample()
            # do action on state
            statevector, self.RL_boolean, self.reward = Simulation().simulate(action, state)
            reward = self.reward

            self.RL_boolean = True
            # TODO: check what info is and define accordingly
            info = None
            self.last_reward = reward
            # set: self.last_action =
            self.episode_total_reward += reward
            done = False
            return statevector, reward, done, info



    def reset(self):
        ''' used after an entire episode and after self.done is set to done'''
        self.last_action = None
        self.last_reward = 0
        self.time = 0
        return self.observation_space.sample()




    def render(self, mode='human', close = False):
        pass






class Simulation(DCEnv):

    def __init__(self):
        # Demand model
        DCEnv.__init__(self)
        self.demandmodel = demandmodel.DemandModel()
        # Reward variables
        self.total_lift_time = None
        self.total_transport_time = None
        self.reward = 0.0







  ##############################################
  #
  # Simulate Demand on Distribution center
  #
  #####################################



  # def generate_supply(self, constant):
  #     ''' we assume that supply is always greater some constant of the demand such that the demand can be served '''
  #     for i in self.demand:
  #         self.supply.append(i+constant)
  #     return
    def simulate(self, action, state):
      ''' simulates applying action to environment and models new state after action is applied to environment'''
      sd = 1.0
      mean = 0.0
      self.demand = self.demandmodel.generate_demand(mean, sd)
      self.total_demand = sum(self.demand)
       # reset variables in the environment to fit action
       # action is a vector of integers defining new size of lift, number of lifts, number of transports, size of transports
      self.lift_size = action[0]
      self.num_lifts = action[1]
      self.num_transports = action[2]
      self.transport_size = action[3]
      # compute reward relevant variables
      self.liftqueue()
      self.lift_times()
      self.transportqueue()
      self.transport_times()

      self.populate_stateVector()
      self.get_reward()
      return self.StateVector, self.RL_boolean, self.reward

    def get_reward(self):
        ''' the greater the reward the worse absolute value is cause penalty storage is negative if demand can't be met by stored supply '''
        #penaltystorage = self.toolittlestorage_penalty()
        reward = -(self.total_transport_time + self.total_lift_time) #+ abs(penaltystorage)#
        self.reward = reward
        return reward



    def populate_stateVector(self):
        ''' new statevector describing the DC
        output: [numlifts, numtransports, sizelifts, sizetransports, queuelift, queuetransport, demand, supply] '''
        self.StateVector = np.array([self.num_transports, self.num_lifts, self.lift_size, self.transport_size, self.transport_queue[0], self.lift_queue[0], self.total_demand, self.total_supply])

  ##############################################
  #
  # generate transport and lift times and queues
  #
  #####################################

    def liftqueue(self):
      ''' we assume lifts can go anywhere for simplicity
      adds demand as queue uniform to the different lifts '''
      alldemand = sum(self.demand)
      uniform_demand = int(alldemand/len(self.lift_queue))
      for i in range(len(self.lift_queue)):
          self.lift_queue[i] = self.lift_queue[i] + uniform_demand
      return

    def transportqueue(self):
      '''we assume transport can go anywhere for simplicity
      adds demand uniform to the different transports'''
      alldemand = sum(self.demand)
      uniform_demand = int(alldemand/len(self.transport_queue))
      for i in range(len(self.transport_queue)):
          self.transport_queue[i] = self.transport_queue[i] + uniform_demand
      return

    def transport_times(self):
      '''upon determining the queue we can estimate the transport times by adding additional time for each product this will be needed for the reward '''
      self.total_transport_time = sum(self.transport_queue) * self.transport_constant
      return

    def lift_times(self):
      '''upon determining the queue we can estimate the transport times by adding additional time for each product this will be needed for the reward '''
      self.total_lift_time = sum(self.lift_queue) * self.lift_constant
      return

      # def toolittlestorage_penalty(self):
      #     '''loss incurred if there is too little storage for the supply such that demand can't be satisified'''
      #     if sum(self.demand) > self.total_elements_storage:
      #         # any difference between the demand and the total elements in storage incurr loss (negative)
      #         loss = self.total_elements_storage - sum(self.demand)
      #         return loss
      #     else:
      #         loss = 0
      #         return loss


    ##############################################
    #
    # RL boolean
    #
    #####################################


    def check_transport_occupancy(self):
        ''' if storage level of any buffer or storage is 2 then action can be set to increase size'''
        if self.transport_occupancy == self.num_transports and self.num_transports != self.max_numtransports:
            # do action RL agent
            self.RL_boolean = True
        if self.lift_occupancy == self.num_lifts and self.num_lifts != self.max_numlifts:
            # do action RL agent
            self.RL_boolean = True



# ##############################################
# #
# # Simulate Demand and supply in Distribution center as well as state of DC
# #
# #####################################
#
#     def additemstorage(self):
#         ''' add items at current time step for supply '''
#         for n in self.supply:
#             if self.current_elements_storage < self.total_elements_storage:
#                 self.current_elements_storage += n
#             else:
#                 pass
#                 print('incurring loss cause no storage')
#
#     def removeitemstorage(self, reduction):
#         ''' how many of the items are removed from storage upon demand is received '''
#         if self.current_elements_storage == 0:
#             self.current_elements_storage = 0
#         for reduction in self.demand:
#             self.current_elements_storage -= reduction
#
#     def setstoragelevel(self):
#         ''' set integer storage level filling of storages '''
#         firstcond = self.current_elements_storage > int(self.total_elements_storage/3)
#         secondcond = self.current_elements_storage < 2*int(self.total_elements_storage/3)
#         if self.current_elements_storage < int(self.total_elements_storage/3):
#             self.storage_level = 0
#         if firstcond and secondcond:
#             self.storage_level = 1
#             # storage full is 3
#         if self.current_elements_storage == self.total_elements_storage:
#             self.storage_level = 3
#         else:
#             self.storage_level = 2
#
#     def set_max_elements_storage(self):
#         '''after RL action changes storage reset the maximum elements in the storage
#         1 item per 10 squarecm '''
#         for i in self.size_storages:
#             self.max_elements_storages += i
#         # set max number of supplies
#         self.total_elements_storage = sum(self.max_elements_storages)

    # def additembuffer(self):
    #     self.current_elements_buffer += 1
    #
    # def removeitembuffer(self):
    #     if self.current_elements_buffer == 0:
    #         self.current_elements_buffer = 0
    #     else:
    #         self.current_elements_buffer -= 1
    #
    # def setbufferlevel(self):
    #     firstcond = self.current_elements_buffer > int(self.max_elements_buffer/3)
    #     secondcond = self.current_elements_buffer < 2*int(self.max_elements_buffer/3)
    #     if self.current_elements_buffer < int(self.max_elements_buffer/3):
    #         self.buffer_level = 0
    #     if firstcond and secondcond:
    #         self.buffer_level = 1
    #     else:
    #         self.buffer_level = 2


######################################
#
# set up distribution center functions i.e. actions of RL agents and
# operations are coded with an integer value and selected with a certain probability (discretized ND in 11 discrete steps)
# resizing operations are applied with a range of 10 sizes again selected according to ND
################################


    # def resize_storage(self):
    #     '''resize storage depends on the number of lifts
    #     one lift will remove a certain area allocated for storage
    #     depends on the number of transports one transport will remove a certain area of total DC'''
    #     for i in self.size_storages:
    #         self.max_elements_storages += i
    #     # set max number of supplies
    #     self.max_elements_storage = sum(self.max_elements_storages)

    # def add_lift(self):
    #     '''adding lifts removes some size of the lift from the total isle size '''
    #     maxi = self.max_numlifts - self.num_lifts
    #     new_lifts = spaces.Discrete(maxi)
    #     new_lift = new_lifts.sample()
    #     self.num_lifts += new_lift
    #     self.policy.append(self.num_lifts)

    # def add_transport(self):
    #     ''' adding transports also requires a resizing of the isle to transport size + constant'''
    #     maxi = self.max_numtransports-self.num_transports
    #     newtransports = spaces.Discrete(maxi)
    #     new_transport = newtransports.sample()
    #     self.num_transports += new_transport
    #     self.policy.append(self.num_transports)

    # def resize_isle(self):
    #     ''' if more transports are added than isles available we need to increase the size of the isles accordingly call whenever transports are adapted'''
    #     if self.num_transports > self.num_isle:
    #         minimum_number_of_isles = self.transports/self.num_isle
    #         minimum_size_of_isles = self.transport_size + self.isle_margin
    #
    # def resize_transport_up(self):
    #     ''' resizing the transport enables a single transport to carry more products  '''
    #     maxi = self.max_size_transport - self.transport_size
    #     size = spaces.Discrete(maxi).sample()
    #     self.policy.append(size)
    #
    # def resize_lift_up(self):
    #     ''' resizing lift enables a single lift to carry more products '''
    #     maxi = self.max_size_lift - self.lift_size
    #     size = spaces.Discrete(maxi).sample()
    #     self.policy.append(size)
    #
    # def resize_transport_down(self):
    #     ''' resizing the transport enables a single transport to carry less products  '''
    #     maxi = self.transport_size - self.product_size
    #     size = spaces.Discrete(maxi).sample()
    #     self.policy.append(size)
    #
    # def resize_lift_down(self):
    #     ''' resizing lift enables a single lift to carry less products '''
    #     maxi = self.lift_size - self.product_size
    #     size = spaces.Discrete(maxi).sample()
    #     self.policy.append(size)
    #
    # def select_operation(self):
    #     ''' select the operation '''
    #     numoperations = 4
    #     operation = spaces.Discrete(maxi).sample()
    #     self.policy.append(operation)
    #
    # def select_operation_sizing(self):
    #     ''' select the operation sizing up or down '''
    #     numoperations = 8
    #     operation = spaces.Discrete(maxi).sample()
    #     self.policy.append(operation)
    #


# ######################################
# #
# # Check if any capacities are reached through spawning of demand
# # in all cases any action might improve the situation e.g. if storage is full another lift could alleviate the situation by contributing to faster transport times
# ################################
#
#     def check_storage_levels(self):
#         ''' if storage level of any buffer or storage is 2 then action can be set to increase size'''
#         if self.storage_level == 2:
#             # do action RL agent
#             self.RL_boolean = True
#         # if self.buffer_level == 2:
#         #     # do action RL agent
#         #     self.RL_boolean = True
