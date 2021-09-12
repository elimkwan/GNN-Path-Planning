from core.DataLoader import *
from core.DataStructure import OccupancyGrid
from environment.VoronoiDirected import VoronoiDirected
from environment.VoronoiDirectedInit import getVoronoiDirectedGraph
from BayesianOptimisation.experiment_setup import Experiment
from planner.CBS import cbs
import core.Constant as constant

import seaborn as sb
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import math

class Simulator:
    def __init__(self, graph, start_nodes, end_nodes, exp=None):
        self.G = graph
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.subgraph_thres = None
        self.cutoff = None
        self.exp = exp

    def get_cutoff(self):
        # Only optimiser edges with length above a certain threshold
        graph = self.G
        distance = []
        assigned = {}
        for n in graph.nodes:
            for neighbor in graph.neighbors(n):
                if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                    d = graph.edges[n, neighbor, 0]['capacity']
                    distance.append(d)
                    assigned[frozenset((n, neighbor))] = 1

        samples_to_be_considered = round(len(distance) * (constant.CONSTRAIN_PROBLEM/100))
        distance = sorted(distance, reverse=True)
        print("total number of distance", len(distance))
        # cutoff0 = distance[samples_to_be_considered]
        cutoff0 = 2

        # reversing the list
        dist = list(reversed(distance))
        # finding the index of element
        index = dist.index(cutoff0)
        # printing the final index
        final_index = len(dist) - index - 1

        self.cutoff = cutoff0
        return cutoff0, final_index

    def get_usage(self, x, a, b):
        # mse = ((num_connecting_nodes - 0.5)**2 + (1-num_connecting_nodes-0.5)**2)/2
        # rmse = math.sqrt(mse)
        # use = m*rmse + c
        # use = a*np.log10(x)/np.log10(base)
        use = (1+a*np.exp(-x+b))**(-1)
        return use

    def updateEdgeProbability(self, graph, probability):
        # Only optimiser edges with length above a certain threshold
        i = 0
        assigned = {}
        idx = int(len(probability)-3)
        direction_p = probability[:idx]
        # used_p = self.get_usage(direction_p, probability[-3], probability[-2])
        for n in graph.nodes:
            for neighbor in graph.neighbors(n):

                # add condition that edge is not linked to start/end loca, for generating dataset for neural network
                # if np.any(self.start_nodes == n) or \
                # np.any(self.end_nodes == n) or \
                # np.any(self.start_nodes == neighbor) or \
                # np.any(self.end_nodes == neighbor):
                #     tail_segment = True
                # else: 
                #     tail_segment = False

                if n != neighbor and frozenset((n, neighbor)) not in assigned.keys():
                    d = graph.edges[n, neighbor, 0]['capacity']
                    if d >= self.cutoff and i < len(direction_p):
                        a1 = len([n for n in graph.neighbors(n)])
                        a2 = len([n for n in graph.neighbors(neighbor)])
                        num_connecting_nodes = a1+a2
                        usage = self.get_usage(num_connecting_nodes, probability[-3], probability[-2])
                        graph.edges[n, neighbor, 0]['probability'] = (usage)*direction_p[i]
                        graph.edges[neighbor, n, 0]['probability'] = (usage)*(1-direction_p[i])
                        assigned[frozenset((n, neighbor))] = 1
                        i += 1
                        # print("Fat Edge Probability", graph.edges[n, neighbor, 0]['probability'], graph.edges[neighbor, n, 0]['probability'])

        return i

    def getGraph(self):
        return self.G

    def run_simulator(self, probability, return_paths = False):
        start_locations = self.start_nodes
        end_locations = self.end_nodes
        # print("Probability", probability)
        self.updateEdgeProbability(self.G, np.array(probability))
        self.subgraph_thres = probability[-1]
        directed_voronoi = VoronoiDirected(self.G, exp=self.exp)
        directed_voronoi_sub = VoronoiDirected(self.G, exp=self.exp)
        env = None

        if return_paths:
            # Clean G if printing ending solution
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)
            
            # if cbs_out == None:
                # print("Cant find solution with SubGraph, return high cost")
                # cbs_out = cbs(directed_voronoi, start_locations, end_locations)
                # paths, cost = cbs_out
                # global_cost, ft, ut = directed_voronoi.getOptimiser     `Cost(paths)
                # return paths, global_cost, subgraph, ft, ut, directed_voronoi_sub, 0

            # print("Path", paths)

            global_cost, ft, ut, penality, conwait, u2 = directed_voronoi_sub.getOptimiserCost(paths, cost, self.exp)
            if paths == None:
                paths = np.array(self.start_nodes).reshape((len(self.start_nodes), -1))
            # print("cost", global_cost)
            return paths, global_cost, subgraph, ft, ut, directed_voronoi, self.subgraph_thres, penality, conwait, u2
        else:    
            subgraph = directed_voronoi_sub.formSubGraph(
                thres=self.subgraph_thres, 
                start_nodes = self.start_nodes,
                end_nodes = self.end_nodes)
            paths, cost = cbs(directed_voronoi_sub, start_locations, end_locations)
            # if cbs_out == None:
                # print("Cant find solution with subgraph, return high cost")
                # return 100000

            global_cost, ft, ut, penality, conwait, u2 = directed_voronoi_sub.getOptimiserCost(paths, cost, self.exp)
            # print("cost", global_cost)
            return global_cost

def generate_data(simulator = None, iterations = 100, dataset = 1, num_agent=1):

    x_path = "./data/ANN/TrainingData/Dataset" + str(dataset) + "-Agent" + str(num_agent) + "_x_path.csv"
    y_path = "./data/ANN/TrainingData/Dataset" + str(dataset) + "-Agent" + str(num_agent) + "_y_path.csv"
    # y_path = "./data/ANN/Samples"+ str(iterations) + "Dataset" + str(dataset) + "_y_path.csv"

    print("Saving Data to", x_path)

    cutoff_thres, num_probabilities = simulator.get_cutoff()
    # print("Number of trainable probabilities", num_probabilities)
    # print("Length cutoff threshold", cutoff_thres)

    for i in range(iterations):
        # print("\n Generating Data, Sample ", i)

        unused_ub = 1
        subgraph_ub = constant.MAX_SUBGRAPH #0.05 
        init_probability = np.zeros((num_probabilities+2+1))
        init_probability[:num_probabilities] = np.random.random_sample(num_probabilities) #direction
        # init_probability[num_probabilities:-1] = np.random.random_sample(num_probabilities) #direction
        # init_probability[-3] = (0.5-(-0.5)) * np.random.random_sample() + (-0.5) # m
        # init_probability[-2] = (0.75-(0.25)) * np.random.random_sample() + (0.25) # c
        init_probability[-3] = (10-(0)) * np.random.random_sample() + (0) # m
        init_probability[-2] = (5-(0.1)) * np.random.random_sample() + (0.1) # c
        # init_probability[-2] = 0.5
        init_probability[-1] = (subgraph_ub - 0) * np.random.random_sample() + 0
        # print("m",init_probability[-3], "c", init_probability[-2], "sub", init_probability[-1])
        output = simulator.run_simulator(
            init_probability, 
            return_paths=True)

        try:
            p, global_cost, _, ft, ut, _, _ , _, conwait, u2 = output
            p = np.array(p)
            arr = np.repeat(i, p.shape[0]).reshape((p.shape[0],1))
            paths = np.hstack((arr , p))
            # print("global cost", global_cost)

            if i == 0:
                with open(x_path, 'a+') as f:
                    pd.DataFrame(init_probability).to_csv(f, index=None)
                with open(y_path, 'a+') as f:
                    pd.DataFrame([global_cost]).to_csv(f, index=None)
            else:
                with open(x_path, 'a+') as f:
                    pd.DataFrame(init_probability).to_csv(f, index=None, header=None)
                with open(y_path, 'a+') as f:
                    pd.DataFrame([global_cost]).to_csv(f, index=None, header=None)
        except: 
            i-= 1

    return


def ann_voronoi_directed(exp, dataset = 1, num_sample = 100):

    if not exp.initialised:
        exp.setParameters()

    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    simulateObj = Simulator(G, exp.start_nodes, exp.end_nodes, exp=exp)

    generate_data(simulator = simulateObj, dataset = dataset, iterations = num_sample, num_agent = exp.NUM_OF_AGENT)

    return


def finalRun(
    probability = None,
    G = None,
    start_nodes = None,
    end_nodes = None,
    exp = None):

    sim = Simulator(G, start_nodes, end_nodes)
    cutoff_thres, num_probabilities = sim.get_cutoff()
    print("Number of trainable probabilities", num_probabilities)
    print("Length cutoff threshold", cutoff_thres)

    sim.updateEdgeProbability(G, probability)
    G = sim.getGraph()

    subgraph_thres = probability[-1]
    directed_voronoi = VoronoiDirected(G, exp=exp)
    directed_voronoi_sub = VoronoiDirected(G, exp=exp)
    env = None

    # Clean Graph when printing ending solution
    subgraph = directed_voronoi_sub.formSubGraph(
        thres = subgraph_thres, 
        start_nodes = start_nodes,
        end_nodes = end_nodes)

    paths, cost = cbs(directed_voronoi_sub, start_nodes, end_nodes)

    if paths == None:
        print("Cant find complete solution with SubGraph")
        paths = np.array(start_nodes).reshape((len(start_nodes), -1))
        
    return paths, cost, subgraph, directed_voronoi_sub, subgraph_thres



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

    paths, cost, subgraph, env, subgraph_thres = finalRun(
        probability = opt_probabilities,
        G = init_graph,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes, 
        exp=exp)
    
    paths_np = np.array(paths)
    if np.any(paths_np[:,-1] != exp.end_nodes):
        print("\nCannot find complete solution. Some didnt reach goal\n")

    global_cost, ft, u1, penality, conwait, u2 = env.getOptimiserCost(paths, cost, exp)

    # ut2 = env.getCoverage(exp)
    # u2 = 0
    congestion, maxmax, avgavg = env.getCongestionLv(paths=paths)

    return paths, global_cost, ft, u1, u2, conwait, maxmax, avgavg, init_graph, subgraph, subgraph_thres, penality 
