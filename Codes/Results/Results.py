from Codes.Visualization.ResultsVisualization import ResultsVisualization
from Codes.Configuration import Result_Path, Methods, TEST_STAGE
from Codes.SumoGraph import Graph
import pandas as pd
import os

class Results:
    def __init__(self, graph :Graph):
        self.graph = graph
        self.Results_visualization = ResultsVisualization()
    
    def RL_results(self, method_name, junction_id):
        if not method_name in [Methods.Fixed, Methods.Maximum_Based, Methods.Random]:
            self.Results_visualization.Results_plot(junction_id, [self.graph.results_history.reward_history_per_episode[junction_id]], ["Reward", "Episode"], "Episode", "Reward", str(method_name)+' Results', 1, TEST_STAGE)
            self.Results_visualization.Results_plot(junction_id, [self.graph.results_history.reward_history_per_episode[junction_id], ], ["Reward", "Episode"], "Episode", "Reward", str(method_name)+' Results', 1, TEST_STAGE)
        self.Results_visualization.Results_plot(junction_id, [self.graph.results_history.waiting_time_history_per_episode[junction_id]], 
                                                ["Waiting Time"], "Episode", "Waiting Time", str(method_name)+' Results', 1, TEST_STAGE)
        self.Results_visualization.Results_plot(junction_id, [self.graph.results_history.std_waiting_time_history_per_episode[junction_id]], 
                                                ["STD Waiting"], "Episode", "STD Waiting Time", str(method_name)+' Results', 1, TEST_STAGE)
        # self.Results_visualization.Results_plot(junction_id, [self.graph.results_history.density_history_per_episode[junction_id]], 
        #                                         ["Density"], "Episode", "Density", str(method_name)+' Results', 1, TEST_STAGE)
    
    def prepare_all_results(self, method_name, scenario_type):
        self.graph.results_history.save_results_as_CSV(method_name, scenario_type)
        self.graph.results_history.make_results_per_episode(method_name)
        junction_ids = list(self.graph.results_history.reward_history.keys())
        for junction_id in junction_ids:
            self.RL_results(method_name, junction_id)