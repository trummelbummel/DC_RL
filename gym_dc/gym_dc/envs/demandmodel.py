
import numpy as np

class DemandModel():

    def __init__(self):
        self.demandaverage_rawdata = None
        self.months = None
        self.filename = None



    def read_data(self):
        data = pd.read_csv(filename)
        self.demandaverage_rawdata = data['demand']
        self.months = data['months']


    def sample_demand_ND(self, mean, month):
        # draw samples from a normal distribution
        sample = np.random.normal(mean)#
        return sample

    def sample_demand_logNormal(self, mean, sd):
        # draw samples from a lognormal distribution, demand usually log normal distributed (ND would sometimes give negative demand)
        # the smaller the standard deviation the more it looks like anormal distribution
        sample = np.random.lognormal(mean, sd)#
        return sample

    def generate_demand(self, mean, sd):
      ''' generates demand for different products 1, 2, ... n from customer side
      n = num products as integer
      sd_i = standard deviation product i , array element
      µ_i = mean for product i , array element'''

      demand = []
      dem = self.sample_demand_logNormal(mean, sd) * 1000
     
      demand.append(dem)
      return demand


    # def safety_stock_threshold(self):
    #     # https://blog.kinaxis.com/2013/02/truth-lies-and-statistical-modeling-in-supply-chain/
    #
