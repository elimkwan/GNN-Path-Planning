from core.DataLoader import *
from core.DataStructure import OccupancyGrid, Point
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
from BayesianOptimisation.sets_of_exp import Experiment_Sets
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

def getTrainableEdges(edges_dict, graph):
    edges = edges_dict.keys()
    trainable_edges = []
    for e in edges:
        p0 = graph.nodes[e[0]]['position']
        p1 = graph.nodes[e[1]]['position']
        trainable_edges.append((
            round(p0.x,2),
            round(p0.y,2),
            round(p1.x,2),
            round(p1.y,2)
        ))
    return trainable_edges

def gen_voronoi_directed(exp, exp_set, first_sample = True, num_sample=100):

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    edges = {}
    for elem in G.edges():
        edge = (elem[0], elem[1])
        edges[edge] = 1

    simulateObj = Simulator(G, exp, exp_set=exp_set, edge_list=edges)
    loc_to_key = simulateObj.loc_to_key

    np.random.RandomState(42)
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias

    domain = []
    k = 0
    assigned = {}
    trainable_edges = []

    # print(simulateObj.G.nodes)
    for e in exp_set.trainable_edges:
        if (e[0],e[1]) in loc_to_key.keys() and (e[2],e[3]) in loc_to_key.keys():
            n1 = loc_to_key[(e[0],e[1])]
            n2 = loc_to_key[(e[2],e[3])]
            if n1 != n2 \
                and frozenset((n1, n2)) not in assigned.keys() \
                and G.has_edge(n1, n2, 0) \
                and G.edges[(n1, n2, 0)]['capacity'] > 1:

                direction_percentage = {
                    'name': k , 
                    'type': 'continuous', 
                    'domain': (0,1)
                }
                domain.append(direction_percentage)
                assigned[frozenset((n1, n2))] = 1
                k += 1

                if first_sample:
                    edge_temp = (round(G.nodes[n1]['position'].x,2), round(G.nodes[n1]['position'].y,2), round(G.nodes[n2]['position'].x,2), round(G.nodes[n2]['position'].y,2))
                    trainable_edges.append(edge_temp)

    print("Number of trainable probabilities", k)


    bo_initial_sample = int(num_sample * 0.6)
    opt = BayesianOptimization(f = simulateObj.run_simulator, maximize=False, \
                                domain=domain, model_type='GP', \
                                initial_design_numdata = bo_initial_sample,\
                                kernel=kern, acquisition_type='MPI')

    out_name="bo-results.txt"
    opt.run_optimization(max_iter=constant.BO_OPT_SAMPLES, report_file=out_name)

    if first_sample:
        #update trainable edges
        probabilities = opt.x_opt
        exp_set.update_trainable_edges_once(trainable_edges)

    else:
        #fill up the array
        probabilities = []
        k = 0
        assigned = {}
        for e in exp_set.trainable_edges:
            if (e[0],e[1]) in loc_to_key.keys() and (e[2],e[3]) in loc_to_key.keys():
                n1 = loc_to_key[(e[0],e[1])]
                n2 = loc_to_key[(e[2],e[3])]
                if n1 != n2 and frozenset((n1, n2)) not in assigned.keys():
                    if simulateObj.G.has_edge(n1, n2, 0):
                        probabilities.append(opt.x_opt[k])
                        assigned[frozenset((n1, n2))] = 1
                        k += 1
                    else:
                        probabilities.append(0.5)
            else:
                probabilities.append(0.5)

    return probabilities, opt, G, exp_set 

# def gen_voronoi_directed(exp, exp_set, first_sample = True, num_sample=100):

#     G = getVoronoiDirectedGraph(
#         occupancy_grid = exp.occupancy_grid,
#         nodes = exp.nodes, 
#         edges = exp.edges_dir,
#         start_nodes = exp.start_nodes,
#         end_nodes = exp.end_nodes)

#     edges = {}
#     for elem in G.edges():
#         edge = (elem[0], elem[1])
#         edges[edge] = 1

    

#     simulateObj = Simulator(G, exp, exp_set=exp_set, edge_list=edges)
#     loc_to_key = simulateObj.loc_to_key

#     np.random.RandomState(42)
#     kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
#     kern_bias = GPy.kern.Bias(input_dim=2)
#     kern = kern_eq + kern_bias

#     domain = []
#     k = 0
#     assigned = {}
#     trainable_edges = []

#     # print(simulateObj.G.nodes)
#     for e in exp_set.trainable_edges:
#         if (e[0],e[1]) in loc_to_key.keys() and (e[2],e[3]) in loc_to_key.keys():
#             n1 = loc_to_key[(e[0],e[1])]
#             n2 = loc_to_key[(e[2],e[3])]
#             if n1 != n2 \
#                 and frozenset((n1, n2)) not in assigned.keys() \
#                 and G.has_edge(n1, n2, 0) \
#                 and G.edges[(n1, n2, 0)]['capacity'] > 1:

#                 direction_percentage = {
#                     'name': k , 
#                     'type': 'continuous', 
#                     'domain': (0,1)
#                 }
#                 domain.append(direction_percentage)
#                 assigned[frozenset((n1, n2))] = 1
#                 k += 1

#                 if first_sample:
#                     edge_temp = (round(G.nodes[n1]['position'].x,2), round(G.nodes[n1]['position'].y,2), round(G.nodes[n2]['position'].x,2), round(G.nodes[n2]['position'].y,2))
#                     trainable_edges.append(edge_temp)

#     print("Number of trainable probabilities", k)


#     bo_initial_sample = int(num_sample * 0.6)
#     opt = BayesianOptimization(f = simulateObj.run_simulator, maximize=False, \
#                                 domain=domain, model_type='GP', \
#                                 initial_design_numdata = bo_initial_sample,\
#                                 kernel=kern, acquisition_type='MPI')

#     out_name="bo-results.txt"
#     opt.run_optimization(max_iter=constant.BO_OPT_SAMPLES, report_file=out_name)

#     if first_sample:
#         #update trainable edges
#         probabilities = opt.x_opt
#         exp_set.update_trainable_edges_once(trainable_edges)

#     else:
#         #fill up the array
#         probabilities = []
#         k = 0
#         assigned = {}
#         for e in exp_set.trainable_edges:
#             if (e[0],e[1]) in loc_to_key.keys() and (e[2],e[3]) in loc_to_key.keys():
#                 n1 = loc_to_key[(e[0],e[1])]
#                 n2 = loc_to_key[(e[2],e[3])]
#                 if n1 != n2 and frozenset((n1, n2)) not in assigned.keys():
#                     if simulateObj.G.has_edge(n1, n2, 0):
#                         probabilities.append(opt.x_opt[k])
#                         assigned[frozenset((n1, n2))] = 1
#                         k += 1
#                     else:
#                         probabilities.append(0.5)
#             else:
#                 probabilities.append(0.5)

#     return probabilities, opt, G, exp_set


def finalRun(
    probability = None,
    G = None,
    exp = None,
    exp_set = None):

    simulateObj = Simulator(G, exp, exp_set)
    # num_probabilities = simulateObj.updateEdgeProbability()
    loc_to_key = simulateObj.loc_to_key
    # print("Number of trainable probabilities", num_probabilities)

    simulateObj.updateEdgeProbability(graph = G, probability=probability)
    G = simulateObj.getGraph()
    directed_voronoi = VoronoiDirected(G, exp=exp)
    paths, cost = cbs(directed_voronoi, exp.start_nodes, exp.end_nodes)

    # if paths == None or np.any(np.array(paths)!= end_nodes):
    if paths == None:
        print("Cant find complete solution with SubGraph")
        paths = np.array(exp.start_nodes).reshape((len(exp.start_nodes), -1))
        
    return paths, cost, directed_voronoi, G



def get_results(probabilities, exp, exp_set):
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

    paths, cost, env, updated_graph = finalRun(
        probability = probabilities,
        G = init_graph,
        exp = exp,
        exp_set = exp_set)
    
    paths_np = np.array(paths)
    if np.any(paths_np[:,-1] != exp.end_nodes):
        print("\nCannot find complete solution. Some didnt reach goal\n") 

    global_cost, ft, u1, penality, conwait, u2 = env.getOptimiserCost(paths, cost, exp)

    # congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, global_cost, ft, conwait, updated_graph


