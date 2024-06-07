import os
import sys
import optparse
sys.path.append(os.path.abspath("."))

from Codes.Configuration import Network_Path, Methods
from Codes.Controller import Controller

from sumolib import checkBinary
from traci import trafficlight
from traci import start
import traci

class SUMO_ENV:
    
    def __init__(self):
        self.intersections = None
    
    def get_Options(self):
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true",
                            default=True, help="run the commandline version of sumo")
        options, _ = opt_parser.parse_args()
        return options
   
    def Starting(self):
        if self.get_Options().nogui:sumoBinary = checkBinary('sumo')
        else: sumoBinary = checkBinary('sumo-gui')
        start([sumoBinary, "-c", Network_Path])
   
    def exit(self):
        traci.close()
        sys.stdout.flush()

    def Max_Log(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Max_Log)
    
    def Max_Tanh(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Max_Tanh)
    
    def Max_Sigmoid(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Max_Sigmoid)
    
    def Max_Derivative(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Max_Derivative_of_waiting_time)
    
    def Avg_Log(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Avg_Log)
    
    def Avg_Tanh(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Avg_Tanh)
    
    def Avg_Sigmoid(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Avg_Sigmoid)
    
    def Avg_Derivative(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Avg_Derivative_of_waiting_time)
    
    def Fixed(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Fixed)

    def Maximum_Based(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Maximum_Based)

    def Random(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Random)
    
    def Communication(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Communication)

    def Emergency(self):
        controller = Controller(self.intersections)
        controller.Run(method_name=Methods.Emergency)

    def Default_Case(self):
        print("Error Running Method")
    
    def Run_Methodes(self):
        switch_dict = {
            # Methods.Max_Log: self.Max_Log,
            Methods.Fixed: self.Fixed,
            # Methods.Maximum_Bacldsed: self.Maximum_Based,
            # Methods.Random: self.Random,
            # Methods.Avg_Log: self.Avg_Log,
            # Methods.Max_Tanh: self.Max_Tanh,
            # Methods.Avg_Tanh: self.Avg_Tanh,  
            # Methods.Max_Derivative_of_waiting_time: self.Max_Derivative,
            # Methods.Avg_Derivative_of_waiting_time: self.Avg_Derivative, 
            # Methods.Max_Sigmoid: self.Max_Sigmoid,
            # Methods.Avg_Sigmoid: self.Avg_Sigmoid,
            }
        for i, methode in enumerate(switch_dict.keys()):
            # if i != 0: break
            self.Starting()
            self.intersections = trafficlight.getIDList()
            case_function = switch_dict.get(methode, self.Default_Case)
            case_function()
            self.exit()
if __name__ == "__main__":
    # try:
    env = SUMO_ENV()
    env.Run_Methodes()
    # except Exception as e:
    #     print(f"An exception of type {type(e).__name__} occurred.")

