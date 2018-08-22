from gym_dc.envs import demandmodel
from gym_dc.envs.distributioncenter_env import DCEnv

class Simulation(DCEnv):

    def __init__(self):
        # Demand model
        super().__init__()
        self.demandmodel = demandmodel.DemandModel()






  ##############################################
  #
  # Simulate Demand on Distribution center
  #
  #####################################


    def generate_demand(self):
      ''' generates demand for different products 1, 2, ... n from customer side
      n = num products as integer
      sd_i = standard deviation product i , array element
      µ_i = mean for product i , array element'''
      sd = [3]
      mean = [100]
      for i in range(len(mean)):
          self.demand.append(self.demandmodel.sample_demand_logNormal(self, mean, sd))
      return

  # def generate_supply(self, constant):
  #     ''' we assume that supply is always greater some constant of the demand such that the demand can be served '''
  #     for i in self.demand:
  #         self.supply.append(i+constant)
  #     return
    def simulate(self):
      self.generate_demand()
      print('simulating step ')
      self.liftqueue()
      self.lift_times()
      self.transportqueue()
      self.transport_times()
      print(self.transport_queue)
      self.populate_stateVector()
      return self.StateVector, self.RL_boolean, self.transport_times, self.lift_times


    def populate_stateVector(self):
      pass





  ##############################################
  #
  # generate transport and lift times and queues
  #
  #####################################

    def liftqueue(self):
      ''' we assume lifts can go anywhere for simplicity
      adds demand as queue uniform to the different lifts '''
      uniform_demand = int(self.demand/len(self.lift_queue))
      for i in range(len(self.lift_queue)):
          self.lift_queue[i] = self.lift_queue[i] + uniform_demand

    def transportqueue(self):
      '''we assume transport can go anywhere for simplicity
      adds demand uniform to the different transports'''
      uniform_demand = int(self.demand/len(self.transport_queue))
      for i in range(len(self.transport_queue)):
          self.transport_queue[i] = self.transport_queue[i] + uniform_demand

    def transport_times(self):
      '''upon determining the queue we can estimate the transport times by adding additional time for each product this will be needed for the reward '''
      self.total_transport_time = sum(self.transport_queue) * self.transport_constant

    def lift_times(self):
      '''upon determining the queue we can estimate the transport times by adding additional time for each product this will be needed for the reward '''
      self.total_lift_time = sum(self.lift_queue) * self.lift_constant

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
