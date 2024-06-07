from Codes.Configuration import episode_time, traffic_light_period, Result_Path, Methods
import pandas as pd
import numpy as np
import os

class ResultsHistory:
    def __init__(self):
        self.reward_history_per_episode, self.reward_history = {}, {}
        self.epsilon_history_per_episode, self.epsilon_history = {}, {}
        self.Q_Table_history, self.state_history = {}, {}
        self.action_history, self.Action_history = {}, {}
        self.Q_history = {}       
        
        self.std_waiting_time_history, self.std_waiting_time_history_per_episode = {}, {}
        self.waiting_time_history, self.waiting_time_history_per_episode = {}, {}
        self.waiting_time_history_for_edge = {}
        self.density_history, self.density_history_per_episode = {}, {}
        
    def actions_save(self, action, edges):
        for i, edge in enumerate(edges):
            if not edge in self.action_history:self.action_history[edge] = [action[i]]
            else:self.action_history[edge].append(action[i])

    def make_results_per_episode(self, methode_name):
        number_of_steps_per_episode = episode_time // traffic_light_period
        for key in self.reward_history.keys():
            for i in range(0, len(self.reward_history[key]), number_of_steps_per_episode):
                if not key in self.reward_history_per_episode.keys():
                    self.reward_history_per_episode[key] = [np.average(self.reward_history[key][i:i + number_of_steps_per_episode])]
                    self.density_history_per_episode[key] = [np.average(self.density_history[key][i:i + number_of_steps_per_episode])]
                    self.waiting_time_history_per_episode[key] = [np.average(self.waiting_time_history[key][i:i + number_of_steps_per_episode])]
                    self.std_waiting_time_history_per_episode[key] = [np.average(self.std_waiting_time_history[key][i:i + number_of_steps_per_episode])]
                    if not methode_name in [Methods.Fixed, Methods.Maximum_Based, Methods.Random]:
                        self.epsilon_history_per_episode[key] = [np.average(self.epsilon_history[key][i:i + number_of_steps_per_episode])]
                else:
                    self.reward_history_per_episode[key].append(np.average(self.reward_history[key][i:i + number_of_steps_per_episode]))
                    self.density_history_per_episode[key].append(np.average(self.density_history[key][i:i + number_of_steps_per_episode]))
                    self.waiting_time_history_per_episode[key].append(np.average(self.waiting_time_history[key][i:i + number_of_steps_per_episode]))
                    self.std_waiting_time_history_per_episode[key].append(np.average(self.std_waiting_time_history[key][i:i + number_of_steps_per_episode]))
                    if not methode_name in [Methods.Fixed, Methods.Maximum_Based, Methods.Random]:
                        self.epsilon_history_per_episode[key].append(np.average(self.epsilon_history[key][i:i + number_of_steps_per_episode]))
            df = pd.DataFrame()
            df['Reward'], df['Waiting Time'] = self.reward_history_per_episode[key], self.waiting_time_history_per_episode[key]
            df['STD_Waiting Time'], df['Density'] = self.std_waiting_time_history_per_episode[key], self.density_history_per_episode[key]
            if not methode_name in [Methods.Fixed, Methods.Maximum_Based, Methods.Random]:
                df['Epsilon'] = self.epsilon_history_per_episode[key]
            save_path = os.path.join(Result_Path, str(methode_name) + ' Results')
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, "Numerical Results Per Episode " + key + ".csv"), index=False)
            
    def save_results_as_CSV(self, method_name, scenario_type):
        save_path = os.path.join(Result_Path, str(method_name) + ' Results')
        os.makedirs(save_path, exist_ok=True)
        for key in self.reward_history.keys():
            df = pd.DataFrame()
            df['Reward'], df['Waiting Time'] = self.reward_history[key], self.waiting_time_history[key]
            df['STD_Waiting Time'], df['Density'] = self.std_waiting_time_history[key], self.density_history[key]
            if not method_name in [Methods.Fixed, Methods.Maximum_Based, Methods.Random]:
                df['State'], df['Action'] = self.state_history[key], self.Action_history[key]
                df['Q_value'], df['waiting for edge'] = self.Q_history[key], self.waiting_time_history_for_edge[key]
                df['Epsilon'], df['Q_Table'] = self.epsilon_history[key], self.Q_Table_history[key]
            df.to_csv(os.path.join(save_path, "Numerical Results Per Step " + key + ".csv"), index=False)