# an environment specific object representing your boservation of the environment

from gym_dc.envs import demandmodel
# from gym_dc.envs import DCEnv

class observation_space():


    def __init__(self):
        pass

        # self.max_width_DC = max_width_DC
        # self.min_width_isle = min_width_isle
        # self.max_length_storage = max_length_storage
        # self.max_width_storage = min_length_storage
        # self.max_elements_storage = max_elements_storage
        # self.max_len_buffer = max_len_buffer
        # self.max_width_buffer = max_width_buffer
        # self.max_elements_buffer = max_elements_buffer
        # self.current_elements_buffer = 0
        # self.current_elements_storage = 0
        # self.buffer_level = None
        # self.storage_level = None
        # self.demand = demandmodel.DemandModel(filename)

    def additemstorage(self):
        self.current_elements_storage += 1

    def removeitemstorage(self):
        if self.current_elements_storage == 0:
            self.current_elements_storage = 0
        self.current_elements_storage -= 1

    def setstoragelevel(self):
        firstcond = self.current_elements_storage > int(self.max_elements_storage/3)
        secondcond = self.current_elements_storage < 2*int(self.max_elements_storage/3)
        if self.current_elements_storage < int(self.max_elements_storage/3):
            self.storage_level = 0
        if firstcond and secondcond:
            self.storage_level = 1
        else:
            self.storage_level = 2

    def additembuffer(self):
        self.current_elements_buffer += 1

    def removeitembuffer(self):
        if self.current_elements_buffer == 0:
            self.current_elements_buffer = 0
        else:
            self.current_elements_buffer -= 1


    def setbufferlevel(self):
        firstcond = self.current_elements_buffer > int(self.max_elements_buffer/3)
        secondcond = self.current_elements_buffer < 2*int(self.max_elements_buffer/3)
        if self.current_elements_buffer < int(self.max_elements_buffer/3):
            self.buffer_level = 0
        if firstcond and secondcond:
            self.buffer_level = 1
        else:
            self.buffer_level = 2
