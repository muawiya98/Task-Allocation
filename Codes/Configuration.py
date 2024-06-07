from Codes.TrafficLightController.TrafficLightActions import Actions
from traci import trafficlight
from enum import Enum
import pickle
import os

# ================= Simulation Settings =================
# Network_Path = "Networks\\environment 3.3\\environment.sumocfg"
# Network_Path = os.path.join(os.path.abspath("."), "Networks", "environment 3.3", "environment.sumocfg")
# Network_Path = "Networks\\environment 2.2\\environment.sumocfg"
# Network_Path = os.path.join(os.path.abspath("."), "Networks", "environment 2.2", "environment.sumocfg")
# Network_Path = "Networks\\environment 1.1\\environment.sumocfg"
Network_Path = os.path.join(os.path.abspath("."), "Networks", "environment 1.1", "environment.sumocfg")

Simulation_Time = 792000

TEST_STAGE = 90

Result_Path = os.path.join(os.path.abspath("."), "Results")

# ================= Traffic Light Settings =================
traffic_light_period = 30

Yellow_period = 5

Green_red_period = 25

# ================= Object Settings =================
generation_period = 450

Vehicle_characteristics = {
    'length': 3,
    'min_cap': 0.5
}

HIGH_NUMBER_OF_VEHICLE = 20

LOW_NUMBER_OF_VEHICLE = 3

# ================= RL Settings =================
episode_time = 7200

NUMBER_OF_ACTION = 6

fully_exploration=25

Weighting_Factor = 0.6
class Methods(Enum):
    Max_Log = 'Max_Log'
    Max_Tanh = 'Max_Tanh'
    Max_Sigmoid = 'Max_Sigmoid'
    Max_Derivative_of_waiting_time = 'Max_Derivative_of_waiting_time'
    Avg_Log = 'Avg_Log'
    Avg_Tanh = 'Avg_Tanh'
    Avg_Sigmoid = 'Avg_Sigmoid'
    Avg_Derivative_of_waiting_time = 'Avg_Derivative_of_waiting_time'
    Fixed = 'Fixed'
    Maximum_Based = 'Maximum_Based'
    Random = 'Random'

# ================= Shared Functions =================
    
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

def get_state_base_action(action, junction_id, emergency_lane, graph):
    programs = trafficlight.getAllProgramLogics(junction_id)
    if action == Actions.N_S_open.value or action == Actions.E_W_open.value:
        phases1 = programs[2].phases
        if action == Actions.N_S_open.value:states = phases1[0].state
        else:states = phases1[2].state
    else:
        phases2 = programs[1].phases
        if action == Actions.N_open.value: states = phases2[0].state
        elif action == Actions.E_open.value: states = phases2[2].state
        elif action == Actions.S_open.value: states = phases2[4].state
        else:states = phases2[6].state
    controlled_lanes = trafficlight.getControlledLanes(junction_id)
    try:
        lane_index = controlled_lanes.index(emergency_lane)
        edge_state = states[lane_index]
        edge_state = graph.lane_state[edge_state]
    except:edge_state = 0
    return edge_state