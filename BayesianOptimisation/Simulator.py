import numpy as np
from environment.VoronoiDirected import VoronoiDirected
from planner.CBS import cbs

# This Simulator Class is mainly maintained for genVoronoiDirected.py 

class Simulator:
    def __init__(self, graph, exp, exp_set=None, edge_list = None):
        self.G = graph
        self.start_nodes = exp.start_nodes
        self.end_nodes = exp.end_nodes
        self.subgraph_thres = None
        self.exp_set = exp_set
        self.exp = exp
        self.edge_list = edge_list

        self.loc_to_key = {} 
        for n in graph.nodes:
            p = graph.nodes[n]['position']
            self.loc_to_key[(round(p.x,2),round(p.y,2))] = n

    def updateEdgeProbability(self, graph=None, probability=None):
        # Only optimiser edges with length above a certain threshold
        i = 0
        assigned = {}
        direction_p = probability

        if self.exp_set != None:
            for e in self.exp_set.trainable_edges:
                if (e[0],e[1]) in self.loc_to_key.keys() and (e[2],e[3]) in self.loc_to_key.keys():
                    n1 = self.loc_to_key[(e[0],e[1])]
                    n2 = self.loc_to_key[(e[2],e[3])]
                    if n1 != n2 \
                        and frozenset((n1, n2)) not in assigned.keys() \
                        and self.G.has_edge(n1, n2, 0) \
                        and self.G.edges[(n1, n2, 0)]['capacity'] > 1:

                        if graph != None :
                            graph.edges[n1, n2, 0]['probability'] = direction_p[i] # Remove Pruning Part
                            graph.edges[n2, n1, 0]['probability'] = 1-direction_p[i] # Remove Pruning Part
                        
                        assigned[frozenset((n1, n2))] = 1
                        i += 1
            return i
        
        elif graph != None:
            for n1 in graph.nodes:
                for n2 in graph.neighbors(n1):
                    if n1 != n2 \
                        and frozenset((n1, n2)) not in assigned.keys() \
                        and graph.edges[(n1, n2, 0)]['capacity'] > 1:

                        graph.edges[n1, n2, 0]['probability'] = direction_p[i] 
                        graph.edges[n2, n1, 0]['probability'] = 1-direction_p[i]
                        assigned[frozenset((n1, n2))] = 1
                        i += 1
            return i
        else:
            print("Error Lack Graph")


    def updateEdgeProbabilityFromEdgeList(self, graph=None, probability=None):
        # Only optimiser edges with length above a certain threshold
        direction_p = probability
        for i, e in enumerate(self.edge_list):
            graph.edges[e[0], e[1], 0]['probability'] = direction_p[i] # Remove Pruning Part
               

    def getGraph(self):
        return self.G

    def run_simulator(self, probabilities, return_paths = False):
        probability = probabilities[0]
        start_locations = self.start_nodes
        end_locations = self.end_nodes
        # self.updateEdgeProbability(graph = self.G, probability = np.array(probability))
        self.updateEdgeProbabilityFromEdgeList(graph = self.G, probability = np.array(probability))
        # self.subgraph_thres = probability[-1]
        directed_voronoi = VoronoiDirected(self.G, exp=self.exp)
        paths, cost = cbs(directed_voronoi, start_locations, end_locations)

        global_cost, ft, ut, penality, conwait, u2 = directed_voronoi.getOptimiserCost(paths, cost, self.exp)
        return global_cost