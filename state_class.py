

class DC_State():
    ''' class for update of state vector representation of DV '''

    def __init__():
        self.statevector = np.array([])


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
        sd = [1, 3]
        mean = [50, 100]
        for i in range(len(mean)):
            self.demand.append(self.demandmodel.sample_demand_logNormal(self, mean, sd, month))
        return

    def generate_supply(self, constant):
        ''' we assume that supply is always greater some constant of the demand such that the demand can be served '''
        for i in self.demand:
            self.supply.append(i+constant)
        return



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

    def toolittlestorage_penalty(self):
        '''loss incurred if there is too little storage for the supply such that demand can't be satisified'''
        if sum(self.demand) > self.total_elements_storage:
            # any difference between the demand and the total elements in storage incurr loss (negative)
            loss = self.total_elements_storage - sum(self.demand)
            return loss
        else:
            loss = 0
            return loss
