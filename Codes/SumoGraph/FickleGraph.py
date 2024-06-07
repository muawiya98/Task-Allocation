from Codes.SumoGraph.ResultsHistory import ResultsHistory
from Codes.SumoGraph.ModelsHistory import ModelsHistory
class FickleGraph:
    def __init__(self, Edge_lane, lane_state, Edge_Junction, all_edges,
                 Junction_Edge, Junction_controlledEdge):
        self.models_history = ModelsHistory(all_edges, Edge_lane, Edge_Junction, lane_state)
        self.Junction_controlledEdge = Junction_controlledEdge
        self.results_history = ResultsHistory()
        self.Edge_Junction = Edge_Junction
        self.Junction_Edge = Junction_Edge
        self.lane_state = lane_state
        self.Edge_lane = Edge_lane
        self.all_edges = all_edges
        self.RL_State = {}