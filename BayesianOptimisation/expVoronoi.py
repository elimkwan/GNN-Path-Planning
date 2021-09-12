from core.DataLoader import *
from core.DataStructure import OccupancyGrid
from environment.Voronoii import Voronoii
from environment.VoronoiiInit import getVoronoiiGraph
from BayesianOptimisation.experiment_setup import Experiment
from planner.CBS import cbs
import core.Constant as constant

import seaborn as sb
import matplotlib.pyplot as plt
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


# def showSolution(result_graph, paths, image, nodes, start_nodes, end_nodes, all_path = True, path_num = 0):
#     edges_in_path = []
#     image2 = 1-image
#     path = paths[path_num]
#     for ite in range(len(path)-1):
#         edges_in_path.append(np.array([path[ite],path[ite+1],0]))
#         p1 = result_graph.nodes[path[ite]]['position']
#         p2 = result_graph.nodes[path[ite+1]]['position']
#         width = result_graph.edges[path[ite],path[ite+1],0]['capacity']
#         distance = result_graph.edges[path[ite],path[ite+1],0]['distance']

#         for r in (p1.x, p2.x, 1):
#             for c in (p1.y, p2.y, 1):
#                 if (Point(r,c) in np.array(nodes)[end_nodes]):
#                     image2[int(r),int(c)] = 1
#                 else:
#                     image2[int(r),int(c)] += 0.2
    
#     drawn = {}
#     fig, ax = plt.subplots(figsize=(12,12))
#     img = np.array(1-image)
#     ax = sb.heatmap(img)
    
#     loop = result_graph.edges if all_path else edges_in_path
    
#     for elem in loop:
#         p1 = result_graph.nodes[elem[0]]['position']
#         p2 = result_graph.nodes[elem[1]]['position']
#         dx = p2.x - p1.x
#         dy = p2.y - p1.y
#         plt.arrow(p1.y, p1.x, dy, dx, head_width = 0.2, alpha=0.5, color = 'grey')
    
#     for p in start_nodes:
#         plt.scatter(nodes[p].y, nodes[p].x, color = 'red', linewidths=5)
#     for p in end_nodes:
#         plt.scatter(nodes[p].y, nodes[p].x, color = 'lime', linewidths=5)

#     plt.gca().invert_yaxis()
#     return plt

def exp_voronoi(exp):

    if not exp.initialised:
        exp.setParameters()

    G = getVoronoiiGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    # print("Obtained Graph")

    voronoi = Voronoii(G, exp=exp)
    paths, cost = cbs(voronoi, exp.start_nodes, exp.end_nodes)
    if paths == None:
        paths = np.array(exp.start_nodes).reshape((exp.NUM_OF_AGENT, -1))

    paths = np.array(paths).reshape((exp.NUM_OF_AGENT, -1))
                
    paths_np = paths
    if np.any(paths_np[:,-1] != exp.end_nodes):
        print("\nCannot find solution\n")

    global_cost, cost_ft, ut, penality, conwait, u2 = voronoi.getOptimiserCost(paths, exp)
    # u2 = 0

    congestion, maxmax, avgavg = voronoi.getCongestionLv(paths=paths)

    return paths, global_cost, penality, cost_ft, ut, u2, conwait, maxmax, avgavg, G
