from core.DataStructure import OccupancyGrid, Point, Edge
from environment.Env import Environment
import core.Constant as constant

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class Voronoii(Environment):
    def __init__(self, graph, exp = None):
        Environment.__init__(self, exp, type = "Voronoi", graph = graph)
        # self.G = graph 
        # self.exp = exp
        # self.sz = exp.image.shape

    def next(self, node, t):
        # returns a list of (next_node, cost) tuples. this represents the children of node at time t.
        x = []

        for neighbor in self.G.neighbors(node):
            d = self.G.edges[node,neighbor]['distance']
            c = self.G.edges[node,neighbor]['capacity']

            # cost = d * (1/(c+0.001)) * (t+1) # Both (Capacity, Distance)
            # cost = d * (t+1) + 0.001*1/(c+0.001)# Distance only
            # cost = 1/(c+0.001) + 0.001*d*(t+1)# Capacity Only

            if self.exp.objective == "Both": 
                cost = d * (1/(0.5*c+0.001))  # Both (Capacity, Direction, Distance)
                # print("Vor-both", cost, 'd', d, 'c', c)
            elif self.exp.objective == "Capacity":
                cost = 1/(0.5*c+0.001) + 0.001*d # Capacity Only
                # print("Vor-cap", cost,  'd', d, 'c', c)
            else:
                cost = d + 0.001*1/(0.5*c+0.001)# Distance only
                # print("Vor-dist", cost, 'd', d, 'c', c)

            x.append([neighbor, cost])
        return x

    def estimate(self, node, goal, t):
        # returns an estimate of the cost to travel from the current node to the goal node
        p1 = self.G.nodes[node]['position']
        p2 = self.G.nodes[goal]['position']
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return (np.linalg.norm(np.array([dx,dy]))+0.0001)

    def getTotalDistanceAndArea(self):
        total_area = 0
        total_dist = 0
        assigned = {}
        for n in self.G.nodes:
            for e in self.G.neighbors(n):
                if n != e and frozenset((n, e)) not in assigned.keys():
                    d = self.G.edges[n,e]['distance']
                    c = self.G.edges[n,e]['capacity']
                    
                    # if ((p1+p2) > thres): # edge (directional) with too low use probability will be eliminated
                    total_area += d * c * (constant.ROBOT_RADIUS*2)
                    total_dist += d
                    assigned[frozenset((n, e))] = 1
                    
        return total_dist, total_area

    def getTravelledDistance(self, paths):
        d = 0
        for path in paths:
            for i in range(len(path)-1):
                d += self.G.edges[path[i],path[i+1]]['distance']
        return d


    def getOptimiserCost(self, solution, exp):
        # Form a Continuous Cost Function
        # Calculate Penality
        last_nodes = []
        manual_set_sol = False
        try:
            for idx, s in enumerate(solution):
                if np.any(s == None):
                    last_nodes.append(exp.start_nodes[idx])
                    manual_set_sol = True
                else:
                    last_nodes.append(s[-1])
        except:
            print("Caught exceptions set lastnode to startnode")
            print("solution", solution)
            last_nodes = exp.start_nodes
            manual_set_sol = True

        aggr = 0
        for i in range(len(last_nodes)):
            end_node = self.G.nodes[exp.end_nodes[i]]['position']
            last_node = self.G.nodes[last_nodes[i]]['position']
            aggr += np.linalg.norm(np.array([last_node.x, last_node.y])-np.array([end_node.x, end_node.y]))
        penality = aggr/constant.NUM_OF_AGENT
            
        if np.all(solution == None) or manual_set_sol:
            travelled_area = 0
            used_roadmap = 0
            total_roadmap = 1
            travelled_dist = exp.NUM_OF_AGENT
            num_of_agent = exp.NUM_OF_AGENT
        else:
            # given a set of paths, find the global cost
            travelled_area = 0
            travelled_dist = 0
            num_of_agent = len(solution)

            # Flow Time
            for agent_path in solution:
                agent_travelled_area = 0
                for idx in range(len(agent_path)-1):
                    cur, nextt = agent_path[idx], agent_path[idx+1]
                    
                    #find the transverse cost between cur and nextt
                    if (cur == nextt):
                        agent_travelled_area += 0
                    else:
                        d = self.G.edges[cur,nextt]['distance']
                        c = self.G.edges[cur,nextt]['capacity']
                        agent_travelled_area += d
                        travelled_dist += d
                travelled_area += agent_travelled_area

            # Roadmap Utilisation
            # total_area = self.getTotalDistance() #tbc
            all_edges = {}
            for e in self.G.edges:
                if e[0] != e[1]:
                    all_edges[frozenset([e[0], e[1]])] = self.G.edges[e[0], e[1]]['distance']*self.G.edges[e[0], e[1]]['capacity']
            # all_edges = {frozenset([e[0], e[1]]):self.G.edges[e[0], e[1]]['distance']*self.G.edges[e[0], e[1]]['capacity'] for e in self.G.edges}
            visited_edge = {}
            used_roadmap = 0
            total_roadmap = 0

            # Calculate total area in roadmap
            for arg_e in all_edges:
                if len(arg_e) < 2: #Dont add capacity of self loop edges
                    continue
                total_roadmap += all_edges[arg_e]
                
            # Calculate used area in roadmap
            # print("len(agent_path)", len(agent_path))
            for time_step in range(len(agent_path)-1):
                transversing_edge = {}
                transversing_edge_length = {}
                for agent_path in solution:
                    cur, nextt = agent_path[time_step], agent_path[time_step+1]
                    if cur != nextt: #waiting doesnt occuiped road
                        if frozenset([cur, nextt]) in transversing_edge:
                            transversing_edge[frozenset([cur, nextt])] += 1
                        else:
                            transversing_edge[frozenset([cur, nextt])] = 1
                            transversing_edge_length[frozenset([cur, nextt])] = self.G.edges[cur,nextt]['distance']
                # print("transversing_edge", transversing_edge)
                for cur_edge in transversing_edge:
                    edge_area = transversing_edge_length[cur_edge]*1
                    if cur_edge in visited_edge and transversing_edge[cur_edge] > visited_edge[cur_edge]:
                        used_roadmap += (transversing_edge[cur_edge] -  visited_edge[cur_edge])*edge_area
                        visited_edge[cur_edge] = transversing_edge[cur_edge]
                    elif cur_edge not in visited_edge:
                        used_roadmap += transversing_edge[cur_edge]*edge_area
                        visited_edge[cur_edge] = transversing_edge[cur_edge]

        ut = used_roadmap/total_roadmap #they are all area
        cost_ut = 1/(ut+0.001)
        cost_ft = travelled_dist/num_of_agent #1+travelled_dist/num_of_agent
        cost_conwait = self.getWaiting(exp = self.exp, paths = solution)
        # global_cost = cost_ft + 2.5 * cost_ut + 100 * (penality)
        if penality < 1:
            penality = 0
        global_cost = cost_ft
        print("cost_ft: ", cost_ft, "ut: ", ut)

        u2 = total_roadmap/exp.free_space

        return global_cost, cost_ft, ut, penality, cost_conwait,u2


    # def getCoverage(self, exp):
    #     total_area = 0
    #     total_dist = 0
    #     assigned = {}
    #     fig, ax = plt.subplots(figsize=(6,6))
    #     plt.xlim(0,34)
    #     plt.ylim(0,34)
    #     count = 0
    #     for n in self.G.nodes:
    #         for e in self.G.neighbors(n):
    #             if n != e and frozenset((n, e)) not in assigned.keys():
    #                 p1 = self.G.nodes[n]['position']
    #                 p2 = self.G.nodes[e]['position']
    #                 d = self.G.edges[n,e]['distance']
    #                 c = self.G.edges[n,e]['capacity']
    #                 assigned[frozenset((n, e))] = 1
                    
    #                 adjustp1 = Point(p1.y, p1.x)
    #                 adjustp2 = Point(p2.y, p2.x)
                    
    #                 refpt1 = adjustp1 if adjustp1.y <= adjustp2.y else adjustp2
    #                 refpt2 = adjustp1 if adjustp1.y > adjustp2.y else adjustp2
                    
    #                 if refpt1.x >= refpt2.x:
    #                     theta_rot = np.pi - np.arctan(abs(refpt1.y - refpt2.y)/abs(refpt1.x - refpt2.x))
    #                 else:
    #                     theta_rot = np.arctan(abs(refpt1.y - refpt2.y)/abs(refpt1.x - refpt2.x))
                        
    #                 if theta_rot >= np.pi/2:
    #                     theta = theta_rot - np.pi/2
    #                 else:
    #                     theta = theta_rot + np.pi/2
                    
    #                 dy = -(c/2)*np.sin(theta)
    #                 if refpt1.y == refpt2.y:
    #                     dx = 0
    #                     width = d
    #                     height = c
    #                     a = 0
    #                 elif refpt1.x > refpt2.x:
    #                     dx = -(c/2)*np.cos(theta)
    #                     width = c
    #                     height = d
    #                     a = (theta) * 180 / np.pi
    #                 elif refpt1.x == refpt2.x:
    #                     dx = -(c/2)
    #                     width = c
    #                     height = d
    #                     a = 0
    #                 else:
    #                     dx = (c/2)*np.cos(np.pi - theta)
    #                     width = d
    #                     height = c
    #                     a = (theta_rot) * 180 / np.pi

    #                 rect = Rectangle((refpt1.x+dx,refpt1.y+dy),width,height,linewidth=0.1,fill=True,angle = a,color = 'black')
    #                 plt.gca().add_patch(rect)

    #     for o in exp.obstacles_loc:
    #         adjustedx, adjustedy = o[1],o[0]
    #         rect = Rectangle((adjustedx-0.5,adjustedy-0.5),1,1,linewidth=0.1,fill=True,angle = 0, color = 'black')
    #         plt.gca().add_patch(rect)
            
    #     ax.axis('off')

    #     im = fig
    #     im.canvas.draw()
    #     X = np.array(im.canvas.renderer._renderer)
    #     X_reshape = X.reshape((-1,4))
    #     X_reshape = np.delete(X_reshape, [1,2,3], axis = 1)
    #     black = np.count_nonzero(X_reshape == 0)
    #     white= np.count_nonzero(X_reshape == 255)
    #     print("Black px", black, "White px", white)
    #     percentage = black/(white+black)
    #     return percentage


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
    #             if paths[a,t] not in self.G.nodes:
    #                 continue
    #             pos = self.G.nodes[paths[a, t]]['position']
    #             congestion[getSubGrid(pos.x),getSubGrid(pos.y, row=False)] += 1
            
    #         acc_congestion.append(congestion)
    #         congestion_flat = congestion.flatten()
    #         maxx.append(np.amax(congestion_flat))
    #         avgg.append(np.average(congestion_flat))

    #     maxx_maxx = np.amax(np.array(maxx))
    #     avgg_avgg = np.average(np.array(avgg))
    #     return acc_congestion, maxx_maxx, avgg_avgg

    # def get_penality_cost(self,solution, start_nodes, end_nodes):
    #     #Form a Continuous Cost Function#
    #     last_nodes = []
    #     manual_set_sol = False
    #     try:
    #         for idx, s in enumerate(solution):
    #             if s == None:
    #                 last_nodes.append(start_nodes[idx])
    #                 manual_set_sol = True
    #             else:
    #                 last_nodes.append(s[-1])
    #     except:
    #         print("Caught exceptions set lastnode to startnode")
    #         last_nodes = start_nodes
    #         manual_set_sol = True

    #     aggr = 0
    #     for i in range(len(last_nodes)):
    #         end_node = self.G.nodes[end_nodes[i]]['position']
    #         last_node = self.G.nodes[last_nodes[i]]['position']
    #         aggr += np.linalg.norm(np.array([last_node.x, last_node.y])-np.array([end_node.x, end_node.y]))

    #     penality = (aggr+1)
    #     return penality

    # def getEdgeCapacity(self, prev_node, node):
    #     if (prev_node,node) in self.G.edges:
    #         c = self.G.edges[prev_node,node]['capacity']
    #     else:
    #         c = 0
    #     return c
    
    # def getNodeCapacity(self, node):
    #     if (node,node) in self.G.edges:
    #         c = self.G.edges[node,node]['capacity']
    #     else:
    #         c = 0
    #     return c

