import numpy as np
from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as constant

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

class Environment:
    def __init__(self, exp, type = "Grid", graph = None):
        self.exp = exp
        self.sz = exp.image.shape
        self.G = graph
        self.type = type

    def next(self):
        print("Returns a list of (next_node, cost) tuples. this represents the children of node at time t.")

    def estimate(self):
        print("Returns an estimate of the cost to travel from the current node to the goal node")

    def getEdgeCapacity(self, prev_node, node):
        # Only for topology based environment
        c = 0
        if self.type == "Voronoi" and (prev_node,node) in self.G.edges:
            c = self.G.edges[prev_node,node]['capacity']
        elif self.type == "VoronoiDirected" and (prev_node,node,0) in self.G.edges:
            c = self.G.edges[prev_node,node,0]['capacity']
        return c
    
    def getNodeCapacity(self, node):
        # Only for topology based environment
        c = 0
        if self.type == "Voronoi" and (node,node) in self.G.edges:
            c = self.G.edges[node,node]['capacity']
        elif self.type == "VoronoiDirected" and (node,node,0) in self.G.edges:
            c = self.G.edges[node,node,0]['capacity']
        return c


    def getOptimiserCost(self):
        print("Return global_cost, cost_ft, ut, penality, cost_conwait, u2")

    def getCoverage(self, exp):
        total_area = 0
        total_dist = 0
        assigned = {}
        fig, ax = plt.subplots(figsize=(6,6))
        plt.xlim(0,34)
        plt.ylim(0,34)
        count = 0
        for n in self.G.nodes:
            for e in self.G.neighbors(n):
                if n != e and frozenset((n, e)) not in assigned.keys():
                    p1 = self.G.nodes[n]['position']
                    p2 = self.G.nodes[e]['position']
                    if self.type == "Voronoi":
                        d = self.G.edges[n,e]['distance']
                        c = self.G.edges[n,e]['capacity']
                    elif self.type == "VoronoiDirected":
                        d = self.G.edges[n,e,0]['distance']
                        c = self.G.edges[n,e,0]['capacity']

                    assigned[frozenset((n, e))] = 1
                    
                    adjustp1 = Point(p1.y, p1.x)
                    adjustp2 = Point(p2.y, p2.x)
                    
                    refpt1 = adjustp1 if adjustp1.y <= adjustp2.y else adjustp2
                    refpt2 = adjustp1 if adjustp1.y > adjustp2.y else adjustp2
                    
                    if refpt1.x >= refpt2.x:
                        theta_rot = np.pi - np.arctan(abs(refpt1.y - refpt2.y)/abs(refpt1.x - refpt2.x))
                    else:
                        theta_rot = np.arctan(abs(refpt1.y - refpt2.y)/abs(refpt1.x - refpt2.x))
                        
                    if theta_rot >= np.pi/2:
                        theta = theta_rot - np.pi/2
                    else:
                        theta = theta_rot + np.pi/2
                    
                    dy = -(c/2)*np.sin(theta)
                    if refpt1.y == refpt2.y:
                        dx = 0
                        width = d
                        height = c
                        a = 0
                    elif refpt1.x > refpt2.x:
                        dx = -(c/2)*np.cos(theta)
                        width = c
                        height = d
                        a = (theta) * 180 / np.pi
                    elif refpt1.x == refpt2.x:
                        dx = -(c/2)
                        width = c
                        height = d
                        a = 0
                    else:
                        dx = (c/2)*np.cos(np.pi - theta)
                        width = d
                        height = c
                        a = (theta_rot) * 180 / np.pi

                    rect = Rectangle((refpt1.x+dx,refpt1.y+dy),width,height,linewidth=0.1,fill=True,angle = a,color = 'black')
                    plt.gca().add_patch(rect)

        for o in exp.obstacles_loc:
            adjustedx, adjustedy = o[1],o[0]
            rect = Rectangle((adjustedx-0.5,adjustedy-0.5),1,1,linewidth=0.1,fill=True,angle = 0, color = 'black')
            plt.gca().add_patch(rect)
            
        ax.axis('off')

        im = fig
        im.canvas.draw()
        X = np.array(im.canvas.renderer._renderer)
        X_reshape = X.reshape((-1,4))
        X_reshape = np.delete(X_reshape, [1,2,3], axis = 1)
        black = np.count_nonzero(X_reshape == 0)
        white= np.count_nonzero(X_reshape == 255)
        print("Black px", black, "White px", white)

        percentage = black/(white+black)
        return percentage

    def getCongestionLv(self, paths=None):
        def getSubGrid(loc, row=True):
            idx = 0 if row else 1
            if loc == 0:
                return 0
            elif loc > self.sz[idx]-2:
                return -1
            else:
                return int((loc-1)/constant.REGION)

        sz = (int(self.sz[0]/constant.REGION), int(self.sz[1]/constant.REGION))
        acc_congestion = []
        total_time = np.array(paths).shape[1]
        agents = np.array(paths).shape[0]
        avgg = []
        maxx = []
        for t in range(total_time):
            congestion = np.zeros(sz)
            for a in range(agents):
                if self.type == "Grid":
                    pos = paths[a][t]
                    congestion[getSubGrid(pos[0]),getSubGrid(pos[1], row=False)] += 1
                elif self.type == "Voronoi" or self.type == "VoronoiDirected":
                    if paths[a,t] not in self.G.nodes:
                        continue
                    pos = self.G.nodes[paths[a, t]]['position']
                    congestion[getSubGrid(pos.x),getSubGrid(pos.y, row=False)] += 1


            acc_congestion.append(congestion)
            congestion_flat = congestion.flatten()
            maxx.append(np.amax(congestion_flat))
            avgg.append(np.average(congestion_flat))

        maxx_maxx = np.amax(np.array(maxx))
        avgg_avgg = np.average(np.array(avgg))
        return acc_congestion, maxx_maxx, avgg_avgg

    def getWaiting(self, exp = None, paths=None, grid = False):
        # Matrice for congestion
        # Share among all environment parameterisation scheme
        if (np.all(paths == None) or np.array(paths).size < exp.NUM_OF_AGENT) and not grid:
            paths = np.array(exp.start_nodes).reshape((exp.NUM_OF_AGENT,1))
        elif (np.array(paths).shape == (exp.NUM_OF_AGENT,) and paths == [None]*exp.NUM_OF_AGENT):
            #[None, None, None, None, None, None, None, None, None, None, None, None]
            paths = np.array(exp.start_nodes).reshape((exp.NUM_OF_AGENT,1))
        elif np.any(paths == None) and not grid:
            try:
                for idx, s in enumerate(paths):
                    if s == None:
                        paths[idx] = [exp.start_nodes[idx]]
            except:
                print("Caught exceptions in getWaiting")
                paths = np.array(exp.start_nodes).reshape((12,1))


        if grid:
            end = [(elem[0], elem[1])for elem in exp.end_locations]
            count = 0
            for a, path in enumerate(paths):
                last_step = None
                for t, step in enumerate(path):
                    initialised = not np.any(last_step == None)
                    not_destination = (step[0], step[1]) not in end
                    if initialised and not_destination and step == last_step: 
                        count += 1
                    last_step = step
            count /= exp.NUM_OF_AGENT     
            return count
        
        count = 0
        # print("paths", paths)
        for a, path in enumerate(paths):
            last_step = None
            for t, step in enumerate(path):
                initialised = not np.any(last_step == None)
                not_destination = step not in exp.end_locations
                if step not in exp.end_nodes:
                    count += 1
                last_step = step
        count /= exp.NUM_OF_AGENT     
        return count
