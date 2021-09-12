from core.DataStructure import OccupancyGrid, Point
from environment.Env import Environment
import core.Constant as constant

import numpy as np
import networkx as nx

class Grid(Environment):
    def __init__(self, exp):
        self.sz = exp.image.shape
        G = nx.grid_2d_graph(self.sz[0],self.sz[1]) 
        for obs in exp.obstacles_loc:
            idx = [int(obs[0]), int(obs[1])]
            n = (idx[0], idx[1])
            if n not in G.nodes:
                continue
            to_be_removed = [(n, nn) for nn in G.neighbors(n)]
            G.remove_edges_from(to_be_removed)
        for e in G.edges:
            G.edges[e]["visited"]=False
        Environment.__init__(self, exp, type = "Grid", graph = G)

        self.total_roadmap = len(self.G.edges)

    def next(self, node, t):
        next_pos = np.array([nn for nn in self.G.neighbors(node)])
        return list(zip(map(tuple,next_pos), np.ones(next_pos.shape[0])))

    def estimate(self, node1, node2, t):
        return np.linalg.norm(np.array(node1)-np.array(node2))

    def getOptimiserCost(self, solution, exp):
        penality = self.getPenalityCost(solution, exp.start_locations, exp.end_locations)

        # Compute total area of roadmap
        ut = 0
        used_roadmap = 0
        G = self.G

        # Compute total area of used roadmap and Flowtime
        num_of_agent = exp.NUM_OF_AGENT
        dist_travelled = 0
        for a, path in enumerate(solution):
            for t, pos in enumerate(path):
                if t == 0 :
                    continue
                if np.all(pos == exp.end_locations[a]):
                    break
                
                past_pos = (int(path[t-1][0]),int(path[t-1][1]))
                if (past_pos,pos) in G.edges:
                    dist_travelled += 1
                    G.edges[(past_pos,pos)]["visited"] = True

        for e in G.edges:
            if G.edges[e]["visited"]:
                used_roadmap += 1

        ut = used_roadmap/self.total_roadmap
        # print("Used roadmap", used_roadmap)
        # print("Total roadmap", self.total_roadmap)
        cost_ft = dist_travelled/num_of_agent
        cost_ut = 1/(ut+0.001)
        # global_cost = cost_ft * cost_ut * (penality+1)
        cost_conwait = self.getWaiting(exp = self.exp, paths = solution, grid=True)
        # global_cost =  cost_ft * cost_ut * (cost_conwait+1) + 1000 * (penality+1)
        global_cost =  cost_ft * cost_ut + 1000 * (penality+1)

        return global_cost, cost_ft, ut, penality, cost_conwait 


    def getPenalityCost(self, solution, start_loc, end_loc):
        aggr = 0
        for i in range(len(start_loc)):
            a = [solution[i][-1][0],solution[i][-1][1]]
            aggr += np.linalg.norm(np.array(a)-np.array(end_loc[i]))

        penality = (aggr)
        return penality


    # def getCongestionLv(self, paths=None):
    #     def getSubGrid(loc, row=True):
    #         idx = 0 if row else 1
    #         if loc == 0:
    #             return 0
    #         elif loc > self.sz[idx]-2:
    #             return -1
    #         else:
    #             return int((loc-1)/constant.REGION)

    #     sz = (int(self.sz[0]/constant.REGION), int(self.sz[1]/constant.REGION))
    #     acc_congestion = []
    #     total_time = np.array(paths).shape[1]
    #     agents = np.array(paths).shape[0]
    #     avgg = []
    #     maxx = []
    #     for t in range(total_time):
    #         congestion = np.zeros(sz)
    #         for a in range(agents):
    #             pos = paths[a][t]
    #             congestion[getSubGrid(pos[0]),getSubGrid(pos[1], row=False)] += 1
            
    #         acc_congestion.append(congestion)
    #         congestion_flat = congestion.flatten()
    #         maxx.append(np.amax(congestion_flat))
    #         avgg.append(np.average(congestion_flat))

    #     maxx_maxx = np.amax(np.array(maxx))
    #     avgg_avgg = np.average(np.array(avgg))
    #     return acc_congestion, maxx_maxx, avgg_avgg



            


