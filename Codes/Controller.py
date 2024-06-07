from Codes.OptimazationUnit.TaskAllocation import nodes_generation_model, tasks_genration_model, simulated_annealing_main, random_main, greedy_main, single_simulated_annealing_main, make_folder
from Codes.Configuration import Network_Path, traffic_light_period, \
    generation_period, episode_time, Simulation_Time, TEST_STAGE, Result_Path, Methods, get_state_base_action, NUMBER_OF_ACTION
from Codes.TrafficLightController.TrafficLightsControler import TrafficLightsController
from Codes.RLUnit.QLearning_Algorithm.Reward import Reward as Qlearning_Reward
from Codes.ObjectsController.SumoController import SumoObjectController
from Codes.InformationProvider.InformationGeter import Infromation
from warnings import simplefilter, filterwarnings
from Codes.Results.Results import Results
from Codes.SumoGraph.Graph import Graph
from Codes.RLUnit.Agent import Agent
import numpy as np
from traci import trafficlight, edge
import random
import traci
import os


random.seed(42)
np.random.seed(42)
filterwarnings('ignore')
filterwarnings(action='once')
simplefilter('ignore', FutureWarning)

class Controller:
    
    def __init__(self, intersection):
        self.number_of_agent = len(intersection)
        self.tls_controller = TrafficLightsController(intersection)
        self.Agent_ids = intersection
        self.graph = Graph(intersection)
        self.results = Results(self.graph)
        self.information = Infromation(self.Agent_ids)
        self.SumoObject = SumoObjectController(self.graph.incomming_edges, self.graph.outcomming_edges)
        self.reward = Qlearning_Reward(self.information, self.graph, self.Agent_ids)
        self.fixed_action = {key:0 for key in self.Agent_ids}
        self.Agents = []

    def Create_Agents(self):
        for Agent_id in self.Agent_ids:
            self.Agents.append(Agent(Agent_id, self.graph, self.reward))
    
    def Save_Start_State(self):
        # path_start_state = Network_Path
        # path_0 = path_start_state.split('.')[0]
        path_start_state = Network_Path+'_start_state.xml'
        traci.simulation.saveState(path_start_state)
        # traci.simulation.saveState(os.path.join(Network_Path, '_start_state.xml'))
    
    def Load_Start_State(self):
        # path_start_state = Network_Path
        # path_0 = path_start_state.split('.')[0]
        path_start_state = Network_Path+'_start_state.xml'
        traci.simulation.loadState(path_start_state)
        # traci.simulation.loadState(os.path.join(Network_Path, '_start_state.xml'))

    
    def Rest_Sumo(self):
        self.Load_Start_State()
    
    def Maping_Between_agents_junctions(self, actions):
        self.tls_controller.send_actions_tls(actions)
        self.tls_controller.check_tls_cmds()
    
    def Save_Actions_For_Edge(self):
        for i, Agent_id in enumerate(self.Agent_ids):
            edges = self.graph.Junction_controlledEdge[Agent_id]
            list_action = []
            for edge_id in edges:
                lanes = self.graph.Edge_lane[edge_id]
                controlled_lanes = trafficlight.getControlledLanes(Agent_id)
                lane_index = list(controlled_lanes).index(lanes[len(lanes) // 2])
                edge_state = trafficlight.getRedYellowGreenState(Agent_id)[lane_index]
                edge_state = self.graph.lane_state[edge_state]
                list_action.append(edge_state)
            self.graph.results_history.actions_save(list_action, edges)

    def Fixed_Action(self, agent_id):
        if self.fixed_action[agent_id]==6:
            self.fixed_action[agent_id] = 1
            return self.fixed_action[agent_id]
        self.fixed_action[agent_id]+=1
        return self.fixed_action[agent_id]

    def Maximum_Based_Action(self, agent_id):
        edge_ids = self.graph.Junction_controlledEdge[agent_id]
        number_of_vehicles =  [edge.getLastStepVehicleNumber(edge_id) for edge_id in edge_ids]
        edge_id = self.graph.Edge_Lane[edge_ids[number_of_vehicles.index(max(number_of_vehicles))]]
        edge_id = edge_id[len(edge_id)//2]
        lane_id = 0
        for i, action in enumerate([1, 2, 3, 4, 5, 6]):
            return get_state_base_action(action, agent_id, lane_id, self.graph)
        return np.random.choice([*range(1,NUMBER_OF_ACTION+1)])

    def Communication_With_Environment(self, method_name, step, scenario_type, episode_number):
        Actions_dic = {}
        for i, Agent_id in enumerate(self.Agent_ids):
            if method_name is Methods.Fixed:
                Actions_dic[Agent_id] = self.Fixed_Action(Agent_id)
                self.reward.Reward_Function(Agent_id, method_name)
            elif method_name is Methods.Maximum_Based:
                Actions_dic[Agent_id] = self.Maximum_Based_Action(Agent_id)
                self.reward.Reward_Function(Agent_id, method_name)
            elif method_name is Methods.Random:
                Actions_dic[Agent_id] = np.random.choice([*range(1,NUMBER_OF_ACTION+1)])
                self.reward.Reward_Function(Agent_id, method_name)    
            else:
                Actions_dic[Agent_id] = self.Agents[i].get_action(step, scenario_type, method_name, episode_number)
        self.Maping_Between_agents_junctions(Actions_dic)
        self.Save_Actions_For_Edge()

    def Run(self, method_name):
        os.makedirs(Result_Path, exist_ok=True)
        os.makedirs(os.path.join(Result_Path, str(method_name) + " Results"), exist_ok=True)
        step_generation, step, sub_episode_number, episode_number = 0, 0, 0, 0
        print(25*"*", method_name, 25*"*")
        self.Create_Agents()
        scenario_type = "train"
        simulated_annealing_path = os.path.join(Result_Path, "simulated_annealing")
        single_simulated_annealing_path = os.path.join(Result_Path, "single_simulated_annealing")
        random_path = os.path.join(Result_Path, "random")
        greedy_path = os.path.join(Result_Path, "greedy")

        make_folder(Result_Path)
        make_folder(simulated_annealing_path)
        make_folder(single_simulated_annealing_path)
        make_folder(random_path)
        make_folder(greedy_path)

        number_of_nodes = random.randint(5, 25)
        nodes_set = nodes_generation_model(number_of_nodes)
        number_of_tasks = random.randint(10, 30)
        tasks_set = tasks_genration_model(number_of_tasks)
        simulated_annealing_nodes, random_nodes, greedy_nodes, single_simulated_annealing_nodes = nodes_set.copy(), nodes_set.copy(), nodes_set.copy(), nodes_set.copy()
        simulated_annealing_tasks, random_tasks, greedy_tasks, single_simulated_annealing_tasks = tasks_set.copy(), tasks_set.copy(), tasks_set.copy(), tasks_set.copy()
        while step < Simulation_Time:
            traci.simulationStep()
            if episode_number>=TEST_STAGE:scenario_type = "test"
            if step == 0: self.Save_Start_State()
            if step % traffic_light_period == 0:
                simulated_annealing_nodes, simulated_annealing_tasks = simulated_annealing_main(simulated_annealing_nodes, simulated_annealing_tasks, step, simulated_annealing_path)
                random_nodes, random_tasks = random_main(random_nodes, random_tasks, step, random_path)
                greedy_nodes, greedy_tasks = greedy_main(greedy_nodes, greedy_tasks, step, greedy_path)
                single_simulated_annealing_nodes, single_simulated_annealing_tasks= single_simulated_annealing_main(single_simulated_annealing_nodes, single_simulated_annealing_tasks, step, single_simulated_annealing_path)
                self.Communication_With_Environment(method_name, step, scenario_type, episode_number)
            if step_generation % generation_period == 0:
                self.SumoObject.generate_object(sub_episode_number, step)
                sub_episode_number += 1
                step_generation = 0
            if (step+generation_period) % episode_time == 0 and step != 0:
                sub_episode_number = 0
                episode_number += 1
                self.Rest_Sumo()
            step_generation += 1
            step += 1
        self.results.prepare_all_results(method_name, scenario_type)
