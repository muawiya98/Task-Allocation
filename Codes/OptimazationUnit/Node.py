from Codes.OptimazationUnit.Configuration import inital_energy
import numpy as np
import random

class Node:
    def __init__(self, node_id=0):
        # self.node_id = node_id # Node.instance_count
        self.processing_speed = np.random.uniform(30, 100) # (vi) # (Million Cycles Per Second)
        self.average_power_consumption = np.random.uniform(4, 10) # (ei) # (mW)
        self.energy_consumption = inital_energy
        self.computational_capacities = np.random.uniform(600, 2000)
        self.used_computational = 0
        self.is_terminated = False
        self.start_time = 0
        self.tasks = []