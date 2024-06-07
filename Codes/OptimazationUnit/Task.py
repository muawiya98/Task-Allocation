from Codes.OptimazationUnit.Configuration import pre_processing_precentage, inference_precentage, post_processing_precentage, high_priorty, second_priorty, low_priorty
import numpy as np
import random

class SubtTask:
    def __init__(self, priorty:int, computation_load:float, communication_load:float, sub_task_name:str, 
                  sub_task_id:int, task_id=0):
        self.sub_task_name = sub_task_name
        self.priorty = priorty
        self.computation_load = computation_load # (p)
        self.communication_load = communication_load # (l)
        self.is_allocated = False
        self.is_executed = False
        # self.task_id = task_id
        self.sub_task_id = sub_task_id

class Task:
    def __init__(self, task_id=0, priorty=None, task_depend_on=None):
        # self.task_id = task_id
        self.computation_load = np.random.uniform(300, 600) # (p) # KCC(Kilo Clock Cycles)
        self.communication_load = np.random.uniform(500, 800) # (l) # bytes of data
        self.is_executed = False
        self.sub_tasks = [SubtTask(priorty=high_priorty, 
                                   computation_load=self.computation_load*pre_processing_precentage,
                                   communication_load=self.communication_load*pre_processing_precentage,
                                   sub_task_name="pre_processing", sub_task_id=0, task_id=task_id),
                          SubtTask(priorty=second_priorty, 
                                   computation_load=self.computation_load*inference_precentage,
                                   communication_load=self.communication_load*inference_precentage,  
                                   sub_task_name="inference", sub_task_id=1, task_id=task_id),
                          SubtTask(priorty=low_priorty, 
                                   computation_load=self.computation_load*post_processing_precentage,
                                   communication_load=self.communication_load*post_processing_precentage, 
                                   sub_task_name="post_processing", sub_task_id=2, task_id=task_id)]