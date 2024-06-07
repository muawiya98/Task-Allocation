from Codes.OptimazationUnit.Configuration import tqueue, bandwidth, eelec, εamp, maximum_transmission_range, W1, W2, W3
import numpy as np
import random


class CostFunction:
    def __init__(self, number_of_nodes, number_of_tasks):
        self.number_of_tasks = number_of_tasks
        self.number_of_nodes = number_of_nodes
        self.execution_time_matrix = np.zeros((number_of_tasks, number_of_nodes)) # (T) 1) The Task Execution Time
        self.energy_consumption_matrix = np.zeros((number_of_tasks, number_of_nodes)) # (E) 2) The Energy Consumption
        
    def set_matrixes(self, number_of_nodes, number_of_tasks):
        self.number_of_tasks = number_of_tasks
        self.number_of_nodes = number_of_nodes
        self.execution_time_matrix = np.zeros((number_of_tasks, number_of_nodes))
        self.energy_consumption_matrix = np.zeros((number_of_tasks, number_of_nodes))
    
    def set_creatin_value_in_matrixes(self, first_ind, secon_ind):
        self.execution_time_matrix[first_ind][secon_ind] = 0    
        self.energy_consumption_matrix[first_ind][secon_ind] = 0
    
    def expand_matrixes(self):
        self.execution_time_matrix = np.concatenate((self.execution_time_matrix, np.zeros((self.number_of_tasks, 1))), axis=1)
        self.energy_consumption_matrix = np.concatenate((self.energy_consumption_matrix, np.zeros((self.number_of_tasks, 1))), axis=1)        
        self.number_of_nodes+=1
        
    # 1) The Task Execution Time
    def calculate_computation_time(self, node, task):
        computation_time = task.computation_load/node.processing_speed
        return computation_time

    def calculate_communication_time(self, task):
        transmission_time = task.communication_load / bandwidth
        communication_time = transmission_time + tqueue
        return communication_time

    def calculate_execution_time(self, node, node_index, tasks):
        self.execution_time_matrix[:, node_index] = 0
        for sub_task_index in node.tasks: 
            computation_time = self.calculate_computation_time(node, tasks[sub_task_index[0]].sub_tasks[sub_task_index[1]])
            communication_time = self.calculate_communication_time(tasks[sub_task_index[0]].sub_tasks[sub_task_index[1]])
            self.execution_time_matrix[sub_task_index[0]][node_index] += (computation_time + communication_time)
    
    def reduce_execution_time(self, node, node_index, task, task_index):        
            computation_time = self.calculate_computation_time(node, task)
            communication_time = self.calculate_communication_time(task)
            self.execution_time_matrix[task_index][node_index] -= (computation_time + communication_time)
        
    def totale_execution_time(self, nodes, tasks):
        for node_index, node in enumerate(nodes):
            self.calculate_execution_time(node, node_index, tasks)
        return np.sum(self.execution_time_matrix)        

    # 2) The Energy Consumption
    def calculate_computation_energy(self, node, node_index, task_index):
        computation_energy = node.average_power_consumption * self.execution_time_matrix[task_index][node_index]
        return computation_energy

    def calculate_communication_energy(self, task):
        energy_consumption_for_transmitting = (eelec + (εamp * (maximum_transmission_range**2))) * task.communication_load
        energy_consumption_for_receiving = eelec * task.communication_load
        communication_energy = energy_consumption_for_transmitting + energy_consumption_for_receiving
        return communication_energy

    def calculate_energy_consumption(self, node, node_index, tasks):
        self.energy_consumption_matrix[:, node_index] = 0
        for sub_task_index in node.tasks:

            computation_energy = self.calculate_computation_energy(node, node_index, sub_task_index[0])
            communication_energy = self.calculate_communication_energy(tasks[sub_task_index[0]].sub_tasks[sub_task_index[1]])
            self.energy_consumption_matrix[sub_task_index[0]][node_index] += (computation_energy + communication_energy)

        
    def totale_execution_energy(self, nodes, tasks):
        for node_index ,node in enumerate(nodes):
            self.calculate_energy_consumption(node, node_index, tasks)
        return np.sum(self.energy_consumption_matrix)

    # 3) The Energy Distribution
    def calculate_energy_distribution(self, nodes):
        energies = [np.sum(self.energy_consumption_matrix[:, i]) for i, node in enumerate(nodes)]
        mean_energy = np.mean(energies)
        distribution = np.sqrt(np.sum((energies - mean_energy) ** 2) / len(nodes))
        return distribution
    
    # 4) The Energy Distribution
    def calculate_execution_time_distribution(self, nodes):
        execution_times = [np.sum(self.execution_time_matrix[:, i]) for i, node in enumerate(nodes)]
        mean_execution_time = np.mean(execution_times)
        distribution = np.sqrt(np.sum((execution_times - mean_execution_time) ** 2) / len(nodes))
        return distribution

    def objective_function(self, nodes, tasks):
        totale_execution_time = self.totale_execution_time(nodes, tasks)
        totale_execution_energy = self.totale_execution_energy(nodes, tasks)
        calculate_energy_distribution = self.calculate_energy_distribution(nodes)
        return (W1*totale_execution_time) + (W2*totale_execution_energy) + (W3*calculate_energy_distribution)