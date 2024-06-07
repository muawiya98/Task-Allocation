import numpy as np
import random
import pickle
import sys
import os


pre_processing_precentage = 0.30
inference_precentage = 0.50
post_processing_precentage = 0.20
high_priorty, second_priorty, low_priorty = 1, 2, 3
inital_energy = 2 # kJ (energy storage is initialized)
tqueue = 0.1 # queue time in seconds
RF = 2.4 # identical radio frequency (RF) modular which works at the 2.4GHz ISM bandwith
bandwidth = 250 # Kbps (B)
eelec = 50/1e9 # nJ/b;
εamp = 10 #10 # pJ/b/m2 (coefficient_of_transmit_amplifier)
maximum_transmission_range = 0.2 #20
W1, W2, W3 = 0.3, 0.3, 0.4
node_removal_id, node_addition_id = 0, 1
number_of_iterations = 100 # temperature
th_size = 100
time_step = 30

def save_object(obj, filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()

def load_object(filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object