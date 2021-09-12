from core.DataLoader import rawInputToArr, rawSceneToArr, getVoronoiiParam
from core.DataStructure import OccupancyGrid, Point
import core.Constant as constant

import numpy as np
import pandas as pd
import networkx as nx

class Experiment:
    def __init__(self, scenemap=None, scene = None, objective="Both", num_agent = 1):
        self.initialised = False
        self.scenemap = scenemap
        self.scene = scene
        self.objective = objective

        self.start_locations = None
        self.end_locations = None
        self.start_end_pair = None
        self.occupancy_grid = None
        self.image = None

        #for Voronoi Diagram
        self.nodes = None
        self.edges_dir = None
        self.edges = None
        self.start_nodes = None
        self.end_nodes = None
        self.obstacles_loc = None
        self.NUM_OF_AGENT = num_agent
        self.ROBOT_RADIUS = constant.ROBOT_RADIUS

        self.free_space = None

    def setParameters(self, specified_start = None, specified_end = None):
        self.obstacles_loc, self.image = rawInputToArr(self.scenemap)
        resolution = constant.RESOLUTION
        self.occupancy_grid = OccupancyGrid(np.array(self.image), [0,0], resolution)


        # Unit Test. Ensure Edge of Occupancy Grid is right, 
        # works when robot radius = 0.5 resolution = 1
        # only works partially when robot radius = resolution = 0.5 
        for row in range (self.image.shape[0]):
            for col in range (self.image.shape[1]):# till 8
                occupancy = 1 if self.occupancy_grid.isOccupied((row,col)) else 0
                if occupancy == 0 and self.image[row,col] == 1:
                    # print(row, col)
                    self.occupancy_grid.changeOccupiedToFree((row, col))

        if np.any(specified_start == None):
            start_locations, end_locations, self.start_end_pair= rawSceneToArr(scene=self.scene, num_agent=self.NUM_OF_AGENT)
        else:
            start_locations = specified_start
            end_locations = specified_end
            aggr = {}
            for a in range (len(start_locations)):
                aggr[Point(start_locations[a,0], start_locations[a,1])] = Point(end_locations[a,0], end_locations[a,1])
            self.start_end_pair = aggr


        self.nodes, self.edges_dir, self.edges, self.start_nodes, self.end_nodes, skipped, self.occupancy_grid = getVoronoiiParam(
            obstacles_loc = self.obstacles_loc, 
            occupancy_grid = self.occupancy_grid, 
            start_end_pair = self.start_end_pair,
            num_agent = self.NUM_OF_AGENT)

        start_locations_tmp = np.array(start_locations[:(self.NUM_OF_AGENT+len(skipped))])
        end_locations_tmp = np.array(end_locations[:(self.NUM_OF_AGENT+len(skipped))])
        self.start_locations = np.delete(start_locations_tmp, skipped, axis = 0)
        self.end_locations = np.delete(end_locations_tmp, skipped, axis = 0) 

        # cal area of total free space
        sz = self.image.shape
        G = nx.grid_2d_graph(sz[0],sz[1]) 
        for obs in self.obstacles_loc:
            idx = [int(obs[0]), int(obs[1])]
            n = (idx[0], idx[1])
            if n not in G.nodes:
                continue
            to_be_removed = [(n, nn) for nn in G.neighbors(n)]
            G.remove_edges_from(to_be_removed)
        self.free_space = len(G.edges)
        self.initialised = True
        print("Set Hyper Parameters, solving for", len(self.start_locations), "agents")

    def saveCleanDataset(self, path = None):
        coord = np.concatenate((self.start_locations, self.end_locations), axis = 1)
        pd.DataFrame(coord).to_csv(path, index = None)
        print("Saving cleaned dataset to ", path)


    #     start_locations, end_locations, start_end_pair= rawSceneToArr(scene=self.scene, num_agent=self.NUM_OF_AGENT)
    #     self.nodes, self.edges_dir, self.edges, self.start_nodes, self.end_nodes, skipped, self.occupancy_grid = getVoronoiiParam(
    #         obstacles_loc = self.obstacles_loc, 
    #         occupancy_grid = self.occupancy_grid, 
    #         start_end_pair = self.start_end_pair,
    #         num_agent = self.NUM_OF_AGENT)





    # def getWaiting(self, paths=None, grid = False):
    #     # Matrice for congestion
    #     # Share among all environment parameterisation scheme
    #     if (np.all(paths == None) or np.array(paths).size < self.NUM_OF_AGENT) and not grid:
    #         paths = np.array(self.start_nodes).reshape((self.NUM_OF_AGENT,1))
    #     elif (np.array(paths).shape == (self.NUM_OF_AGENT,) and paths == [None]*self.NUM_OF_AGENT):
    #         #[None, None, None, None, None, None, None, None, None, None, None, None]
    #         paths = np.array(self.start_nodes).reshape((self.NUM_OF_AGENT,1))
    #     elif np.any(paths == None) and not grid:
    #         try:
    #             for idx, s in enumerate(paths):
    #                 if s == None:
    #                     paths[idx] = [self.start_nodes[idx]]
    #         except:
    #             print("Caught exceptions in getWaiting")
    #             paths = np.array(self.start_nodes).reshape((12,1))


    #     if grid:
    #         end = [(elem[0], elem[1])for elem in self.end_locations]
    #         count = 0
    #         for a, path in enumerate(paths):
    #             last_step = None
    #             for t, step in enumerate(path):
    #                 initialised = not np.any(last_step == None)
    #                 not_destination = (step[0], step[1]) not in end
    #                 if initialised and not_destination and step == last_step: 
    #                     count += 1
    #                 last_step = step
    #         count /= self.NUM_OF_AGENT     
    #         return count
        
    #     count = 0
    #     # print("paths", paths)
    #     for a, path in enumerate(paths):
    #         last_step = None
    #         for t, step in enumerate(path):
    #             initialised = not np.any(last_step == None)
    #             not_destination = step not in self.end_locations
    #             if step not in self.end_nodes:
    #                 count += 1
    #             last_step = step
    #     count /= self.NUM_OF_AGENT     
    #     return count

