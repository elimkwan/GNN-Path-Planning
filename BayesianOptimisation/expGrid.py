from planner.CBS_single import cbs_single
from core.DataLoader import *
from core.DataStructure import OccupancyGrid, Point
from environment.Grid import Grid
from BayesianOptimisation.experiment_setup import Experiment
import core.Constant as constant

def exp_grid(exp):

    if not exp.initialised:
        exp.setParameters()

    resolution = constant.RESOLUTION
    grid = Grid(exp)
    paths, cost = cbs_single(grid, np.array(exp.start_locations), np.array(exp.end_locations))

    if paths == None:
        paths = np.array(exp.start_location).reshape((exp.NUM_OF_AGENT, -1))

    for a, elem in enumerate(paths):
        if np.any(elem == None):
            paths[a] = [exp.start_locations[a]]

    # paths = np.array(paths).reshape((exp.NUM_OF_AGENT, 2, -1))


    cost,ft, ut, penality, conwait = grid.getOptimiserCost(paths,exp)

    congestion, maxmax, avgavg = grid.getCongestionLv(
        paths=paths,
    )

    # penality = grid.get_penality_cost(paths, exp.start_locations, exp.end_locations)
    # print("Grid Penality", penality)

    return paths, cost, penality, ft, ut, 1, conwait, maxmax, avgavg 