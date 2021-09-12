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



def baseline_voronoi_directed(exp):
    G = getVoronoiDirectedGraph(
        occupancy_grid = exp.occupancy_grid,
        nodes = exp.nodes, 
        edges = exp.edges_dir,
        start_nodes = exp.start_nodes,
        end_nodes = exp.end_nodes)

    # Run CBS once to obtain the probability of being a max edge
    directed_voronoi = VoronoiDirected(G, exp=exp)
    paths, cost = cbs(directed_voronoi, exp.start_nodes, exp.end_nodes)

    global_cost, ft, u1, penality, conwait, u2 = directed_voronoi.getOptimiserCost(paths, cost, exp)

    return ft, u1