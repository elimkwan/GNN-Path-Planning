from re import X
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


def getMaxEdgeProb(paths, G):
    edges = {}
    num_edge = 0
    # for a, path in enumerate(paths):
    #     for t, node in enumerate(path[1:]):
    #         edge = frozenset([path[t-1], path[t]])
    #         edges[edge] = 1

    for elem in G.edges():
        edge = (elem[0], elem[1])
        edges[edge] = 1

    num_edge = len(edges)
    time = len(paths[0])

    edge_idx_in_o = {}
    idx_to_edge = {}
    for i, e in enumerate(edges):
        edge_idx_in_o[e] = i
        idx_to_edge[i] = e

    # print(edge_idx_in_o)
    # print(paths)
    occurrence = np.zeros((num_edge, time+1))
    for a, path in enumerate(paths):
        for t, node in enumerate(path[:-1]):
            edge = (paths[a][t], paths[a][t+1])
            index = edge_idx_in_o[edge]
            occurrence[index, t] += 1
    
    # print("occurrence", occurrence)

    for r, row in enumerate(occurrence):
        e = idx_to_edge[r]
        row_maxima = max(row)

        num_row_maxima = 0
        for val in row:
            if val == row_maxima:
                num_row_maxima += 1 

        edges[e] = num_row_maxima/time

    return edges

def normalisedEdges(edges):
    # normalised value
    factor=1.0/sum(edges.values())
    for k in edges:
        edges[k] = edges[k]*factor
    return edges


def getSubsetOfEdges(edges_dict, graph, exp_set):
    edges = edges_dict.keys()
    new_edges_dict = {}
    probability_mask = []

    loc_to_key = {} 
    for n in graph.nodes:
        p = graph.nodes[n]['position']
        loc_to_key[(round(p.x,2),round(p.y,2))] = n

    for elem in exp_set.trainable_edges:
        prev_len = len(new_edges_dict)
        if (elem[0],elem[1]) in loc_to_key.keys() and (elem[2],elem[3]) in loc_to_key.keys():
            n1 = loc_to_key[(elem[0],elem[1])]
            n2 = loc_to_key[(elem[2],elem[3])]
            if (n1,n2) in edges:
                new_edges_dict[(n1,n2)] = edges_dict[(n1,n2)]

        if len(new_edges_dict) > prev_len:
            probability_mask.append(1)
        else:
            probability_mask.append(0)

    print(len(new_edges_dict))
    print(np.count_nonzero(probability_mask))        
    return new_edges_dict, np.array(probability_mask)

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


def util_voronoi_directed(
    exp, for_gnn=False, first_sample = None, exp_set = None, num_sample=100, scheme=2):

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    # Run CBS once to obtain the probability of being a max edge
    directed_voronoi = VoronoiDirected(G, exp=exp)
    paths, cost = cbs(directed_voronoi, exp.start_nodes, exp.end_nodes)
    edges_costs = getMaxEdgeProb(paths, G)

    # For generalisation
    probability_mask = None
    if for_gnn and not first_sample:
        edges_costs, probability_mask = getSubsetOfEdges(edges_costs, G, exp_set)
    elif for_gnn and first_sample:
        trainable_edges = getTrainableEdges(edges_costs, G)
        probability_mask = np.ones(len(trainable_edges))
        exp_set.update_trainable_edges_once(trainable_edges)

    edges_costs = normalisedEdges(edges_costs)

    simulateObj = Simulator(G, exp, exp_set=None, edge_list=edges_costs)
    np.random.RandomState(42)
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias
    domain = []
    k = 0
    assigned = {}
    
    
    for edge in edges_costs:
        if scheme == 1:
            percentage = {
                'name': k , 
                'type': 'continuous', 
                'domain': (0,1)
            }
            domain.append(percentage)
        else: 
            #scheme == 2:
            lower = edges_costs[edge]-0.25
            lower = np.clip(lower, 0, 1)
            upper = edges_costs[edge]+0.25
            upper = np.clip(upper, 0, 1)
            percentage = {
                'name': k , 
                'type': 'continuous', 
                'domain': (lower, upper)
            }
            domain.append(percentage)

    bo_initial_sample = int(num_sample * 0.6)
    opt = BayesianOptimization(f = simulateObj.run_simulator, maximize=False, \
                                domain=domain, model_type='GP', \
                                initial_design_numdata = bo_initial_sample,\
                                kernel=kern, acquisition_type='MPI')

    opt.run_optimization(max_iter=constant.BO_OPT_SAMPLES)

    probabilities = opt.x_opt
    padded_probabilities = []


    num = np.count_nonzero(probability_mask)
    # print("mask", num, "len", len(probabilities))
    assert np.count_nonzero(probability_mask) == len(probabilities)

    k = 0
    if for_gnn:
        m = np.mean(probabilities)
        for i, binary in enumerate(probability_mask):
            if binary == 1:
                padded_probabilities.append(probabilities[k])
                k += 1
            else:
                padded_probabilities.append(m)
    else:
        padded_probabilities = probabilities

    return padded_probabilities, opt, G, exp_set


def finalRun(
    probability = None,
    G = None,
    exp = None,
    exp_set = None):

    simulateObj = Simulator(G, exp)

    simulateObj.updateEdgeProbability(graph = G, probability=probability)
    G = simulateObj.getGraph()
    directed_voronoi = VoronoiDirected(G, exp=exp)
    paths, cost = cbs(directed_voronoi, exp.start_nodes, exp.end_nodes)

    # if paths == None or np.any(np.array(paths)!= end_nodes):
    if paths == None:
        print("Cant find complete solution with SubGraph")
        paths = np.array(exp.start_nodes).reshape((len(exp.start_nodes), -1))
        
    return paths, cost, directed_voronoi, G



def get_results(probabilities, exp, exp_set=None):
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

    return paths, global_cost, ft, u1, updated_graph


