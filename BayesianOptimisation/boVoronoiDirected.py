from core.DataLoader import *
from core.DataStructure import OccupancyGrid
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
from planner.CBS import cbs
import core.Constant as constant
from BayesianOptimisation.Simulator import Simulator

import seaborn as sb
import matplotlib.pyplot as plt
import math
import GPy
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from GPyOpt.methods import BayesianOptimization

# BO without pruning

# class Simulator:
#     def __init__(self, graph, exp):
#         self.G = graph
#         self.start_nodes = exp.start_nodes
#         self.end_nodes = exp.end_nodes
#         self.subgraph_thres = None
#         self.exp = exp

#     def get_cutoff(self):
#         # Only optimiser edges with length above a certain threshold
#         graph = self.G
#         distance = []
#         assigned = {}
#         for n in graph.nodes:
#             for neighbor in graph.neighbors(n):
#                 if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
#                     d = graph.edges[n, neighbor, 0]['capacity']
#                     distance.append(d)
#                     assigned[frozenset((n, neighbor))] = 1

#         samples_to_be_considered = round(len(distance) * (constant.CONSTRAIN_PROBLEM/100))
#         distance = sorted(distance, reverse=True)
#         cutoff0 = 2

#         # reversing the list
#         dist = list(reversed(distance))
#         # finding the index of element
#         index = dist.index(cutoff0)
#         # printing the final index
#         final_index = len(dist) - index - 1

#         self.cutoff = cutoff0
#         return cutoff0, final_index

#     # def get_usage(self, probabilities, m, c):
#     #     usage = []
#     #     for prob in probabilities:
#     #         mse = ((prob - 0.5)**2 + (1-prob-0.5)**2)/2
#     #         rmse = math.sqrt(mse)
#     #         use = m*rmse + c
#     #         usage.append(use)
#     #     return usage

#     def get_usage(self, x, a, b):
#         use = (1+a*np.exp(-x+b))**(-1)
#         return use

#     def updateEdgeProbability(self, graph, probability):
#         # Only optimiser edges with length above a certain threshold
#         i = 0
#         assigned = {}
#         idx = int(len(probability)-3)
#         direction_p = probability[:idx]
#         # used_p = self.get_usage(direction_p, probability[-3], probability[-2])
#         for n in graph.nodes:
#             for neighbor in graph.neighbors(n):

#                 # add condition that edge is not linked to start/end loca, for generating dataset for neural network
#                 # if np.any(self.start_nodes == n) or \
#                 # np.any(self.end_nodes == n) or \
#                 # np.any(self.start_nodes == neighbor) or \
#                 # np.any(self.end_nodes == neighbor):
#                 #     tail_segment = True
#                 # else: 
#                 #     tail_segment = False

#                 if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
#                     d = graph.edges[n, neighbor, 0]['capacity']
#                     if d >= self.cutoff and i < len(direction_p):
#                         a1 = len([n for n in graph.neighbors(n)])
#                         a2 = len([n for n in graph.neighbors(neighbor)])
#                         num_connecting_nodes = a1+a2
#                         usage = self.get_usage(num_connecting_nodes, probability[-3], probability[-2])
#                         graph.edges[n, neighbor, 0]['probability'] = (usage)*direction_p[i]
#                         graph.edges[neighbor, n, 0]['probability'] = (usage)*(1-direction_p[i])
#                         assigned[frozenset((n, neighbor))] = 1
#                         i += 1
#                         # print("Fat Edge Probability", graph.edges[n, neighbor, 0]['probability'], graph.edges[neighbor, n, 0]['probability'])

#         return i

#     def getGraph(self):
#         return self.G

#     def run_simulator(self, probabilities, return_paths = False):
#         probability = probabilities[0]
#         start_locations = self.start_nodes
#         end_locations = self.end_nodes
#         # print("Probability", probability)
#         self.updateEdgeProbability(self.G, np.array(probability))
#         self.subgraph_thres = probability[-1]
#         directed_voronoi = VoronoiDirected(self.G, exp = self.exp)
#         directed_voronoi_sub = VoronoiDirected(self.G, exp = self.exp)
#         env = None

#         # if return_paths:
#         #     # Clean G if printing ending solution
#         #     subgraph = directed_voronoi_sub.formSubGraph(
#         #         thres=self.subgraph_thres, 
#         #         start_nodes = self.start_nodes,
#         #         end_nodes = self.end_nodes)
#         #     paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)

#         #     global_cost, ft, ut, penality, conwait, u2= directed_voronoi_sub.getOptimiserCost(paths, cost, self.exp)
#         #     if paths == None:
#         #         paths = np.array(self.start_nodes).reshape((len(self.start_nodes), -1))
#         #     return paths, global_cost, subgraph, ft, ut,conwait, directed_voronoi, self.subgraph_thres, penality
#         # else:    
#         subgraph = directed_voronoi_sub.formSubGraph(
#             thres=self.subgraph_thres, 
#             start_nodes = self.start_nodes,
#             end_nodes = self.end_nodes,
#             probability = probability)

#         # paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)
#         # global_cost, ft, ut, penality, conwait, u2 = directed_voronoi_sub.getOptimiserCost(paths, cost, self.exp)
#         paths, cost = cbs(directed_voronoi, start_locations, end_locations)
#         global_cost, ft, ut, penality, conwait, u2 = directed_voronoi.getOptimiserCost(paths, cost, self.exp)
#         return global_cost
        

def bo_voronoi_directed(exp, num_sample = 100):

    if not exp.initialised:
        exp.setParameters()

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    simulateObj = Simulator(G, exp)

    np.random.RandomState(42)
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias

    domain = []
    k = 0
    assigned = {}
    for n in G.nodes:
        for neighbor in G.neighbors(n):
            if n != neighbor \
                and frozenset((n, neighbor)) not in assigned.keys()\
                and G.edges[(n, neighbor, 0)]['capacity'] > 1:

                direction_percentage = {
                    'name': k , 
                    'type': 'continuous', 
                    'domain': (0,1)
                }
                domain.append(direction_percentage)
                assigned[frozenset((n, neighbor))] = 1
                k+=1

    print("Number of trainable probabilities", k)

    bo_initial_sample = int(num_sample * 0.6)
    bo_opt_sample = num_sample - bo_initial_sample
        
    opt = BayesianOptimization(f = simulateObj.run_simulator, maximize=False, \
                                domain=domain, model_type='GP', \
                                initial_design_numdata = bo_initial_sample,\
                                kernel=kern, acquisition_type='MPI', verbosity=True)
    opt.run_optimization(max_iter=bo_opt_sample, verbosity=True)
    return opt, G


def finalRun(
    probability = None,
    G = None,
    start_nodes = None,
    end_nodes = None,
    exp=None):

    sim = Simulator(G, exp)
    sim.updateEdgeProbability(G, probability)
    G = sim.getGraph()
    directed_voronoi = VoronoiDirected(G, exp=exp)

    #no subgraph
    print("Use CBS without Subgraph")
    paths, cost = cbs(directed_voronoi, start_nodes, end_nodes)
    if paths == None:
        print("Cant find complete solution with Graph")
        paths = np.array(start_nodes).reshape((len(start_nodes), -1))

    # the [None, None, None ..] case
    if np.array(paths).size == exp.NUM_OF_AGENT and np.all(np.array(paths).flatten() == [None]*exp.NUM_OF_AGENT):
        print("Cant find complete solution with Graph [None, None, None ..]")
        paths = np.array(start_nodes).reshape((len(start_nodes), -1))

    return paths, cost, G, directed_voronoi




def get_results(opt_probabilities, exp):
    # sim = Simulator(init_graph, start_nodes, end_nodes)
    # sim.updateEdgeProbability(init_graph, opt_probabilities)
    # final_graph = sim.getGraph()
    # directed_voronoi_sub = VoronoiDirected(final_graph)

    if not exp.initialised:
        exp.setParameters()

    init_graph = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    paths, cost, graph, env = finalRun(
        probability = opt_probabilities,
        G = init_graph,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes,
        exp=exp)
    
    paths_np = np.array(paths)
    if np.any(paths_np[:,-1] != exp.end_nodes):
        print("\nCannot find complete solution. Some didnt reach goal\n") 


    global_cost, ft, u1, penality, conwait, u2 = env.getOptimiserCost(paths, cost, exp)
    # congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, global_cost, ft, u1, u2, conwait, init_graph, penality
