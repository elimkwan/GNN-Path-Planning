from core.DataStructure import OccupancyGrid, Point, Edge
import core.Constant as c
from ExplicitCorridorMap.ecm import explicitCorridorMap


from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
# import pandas as pd
import copy
import networkx as nx

def initEdgeCapacity(nodes, edges, occupancy_grid):
    edges_coord = [[e.edge_attr['prev_location'], e.edge_attr['next_location']] for e in edges]
    bending_points = explicitCorridorMap(edges_coord, occupancy_grid)

    for idx, e in enumerate(edges):
        e.setObsLocations(bending_points[idx])
        # print("e line ", edges_coord[idx])
        # print("bending_points[idx] ", bending_points[idx])
        if bending_points[idx] == [None, None, None, None]:
            continue
        prev_l, prev_r, next_l, next_r = bending_points[idx]
        left_capacity = int(min(prev_l[-1], next_l[-1])/(c.ROBOT_RADIUS*2))
        right_capacity = int(min(prev_r[-1], next_r[-1])/(c.ROBOT_RADIUS*2))
        e.setCapacity(left_capacity, right_capacity)


# More overlapping road
# def initEdgeCapacity(nodes, edges, occupancy_grid):
    
#     def getSlope(p1,p2):
#         if (p1.y == p2.y):
#             return 100
#         return (p1.x - p2.x) / ((p1.y - p2.y))
    
#     def shiftLine(p1, p2, max_iter, left = False, expand_along_col = True):
#         shifted_p1 = copy.deepcopy(p1)
#         shifted_p2 = copy.deepcopy(p2)
        
#         #Path width based on Robot Width
#         robot_size = -1*(c.ROBOT_RADIUS*2) if left else 1*(c.ROBOT_RADIUS*2)

#         for offset in range(1,int(max_iter/2)):
#             if (expand_along_col):
#                 shifted_p1.y = p1.y + offset*robot_size
#                 shifted_p2.y = p2.y + offset*robot_size
#             else:
#                 shifted_p1.x = p1.x + offset*robot_size
#                 shifted_p2.x = p2.x + offset*robot_size
            
#             if not occupancy_grid.isValidLine(shifted_p1, shifted_p2):
#                 return (offset-1) if offset > 0 else 0
            
#         return offset
    
#     n = copy.deepcopy(nodes)
#     for idx, edge in enumerate(edges):
        
#         if (edge.prev == edge.next):
#             continue
        
#         p1 = copy.deepcopy(n[edge.prev])
#         p2 = copy.deepcopy(n[edge.next])
#         p1_l = copy.deepcopy(n[edge.prev])
#         p2_l = copy.deepcopy(n[edge.next])

#         m = abs(getSlope(p1,p2))
#         max_iter = occupancy_grid.getRows() if m >= 1 else occupancy_grid.getCols()
#         expand_col = True if m > 1 else False
        
#         right_capacity = shiftLine(p1, p2, max_iter, expand_along_col = expand_col)
#         left_capacity = shiftLine(p1_l, p2_l, max_iter, left = True, expand_along_col = expand_col)
        
#         # total_capacity = left_capacity + right_capacity
#         total_capacity = min(left_capacity,right_capacity)*2
#         if total_capacity == 0:
#             total_capacity = 1
        
#         edge.setCapacity(total_capacity)

# Less overlapping road
# def initEdgeCapacity(nodes, edges, occupancy_grid):
    
#     def getSlope(p1,p2):
#         if (p1.y == p2.y):
#             return 100
#         return (p1.x - p2.x) / ((p1.y - p2.y))
    
#     def shiftLine(p1, p2, max_iter, left = False, expand_along_col = True, grid = None):
#         shifted_p1 = copy.deepcopy(p1)
#         shifted_p2 = copy.deepcopy(p2)
#         cur_p1 = copy.deepcopy(p1)
#         cur_p2 = copy.deepcopy(p2)
#         mid_p1 = copy.deepcopy(p1)
#         mid_p2 = copy.deepcopy(p2)
        
#         #Path width based on Robot Width
#         robot_size = -1*(c.ROBOT_RADIUS*2) if left else 1*(c.ROBOT_RADIUS*2)

#         offset = 1

#         for offset in range(1,int(max_iter/2)):
#             # print("offset")
#             if (expand_along_col):
#                 cur_p1.y = p1.y + (offset-1)*robot_size
#                 cur_p2.y = p2.y + (offset-1)*robot_size
#                 mid_p1.y = p1.y + (offset-0.5)*robot_size
#                 mid_p2.y = p2.y + (offset-0.5)*robot_size
#                 shifted_p1.y = p1.y + offset*robot_size
#                 shifted_p2.y = p2.y + offset*robot_size
#             else:
#                 cur_p1.x = p1.x + (offset-1)*robot_size
#                 cur_p2.x = p2.x + (offset-1)*robot_size
#                 mid_p1.x = p1.x + (offset-0.5)*robot_size
#                 mid_p2.x = p2.x + (offset-0.5)*robot_size
#                 shifted_p1.x = p1.x + offset*robot_size
#                 shifted_p2.x = p2.x + offset*robot_size

#             if not grid.isValidLine(shifted_p1, shifted_p2, tolerance=5):
#                 ans = (offset-1) if offset > 0 else 0
#                 return ans, grid
#             else:
#                 #area that are marked for that edge, set to occupy
#                 grid.set_to_occuiped(cur_p1, cur_p2, shifted_p1, shifted_p2, mid_p1, mid_p2)

#         # print("Cant expand")
#         return offset, grid
    
#     n = copy.deepcopy(nodes)
#     new_grid = copy.deepcopy(occupancy_grid)

#     for idx, edge in enumerate(edges):
#         if (edge.prev == edge.next):
#             continue
#         p1 = copy.deepcopy(n[edge.prev])
#         p2 = copy.deepcopy(n[edge.next])
#         p1_l = copy.deepcopy(n[edge.prev])
#         p2_l = copy.deepcopy(n[edge.next])

#         m = abs(getSlope(p1,p2))
#         max_iter = new_grid.getRows() if m >= 1 else new_grid.getCols()
#         # max_iter = 4
#         expand_col = True if m > 1 else False
        
#         right_capacity, new_grid = shiftLine(p1, p2, max_iter, expand_along_col = expand_col, grid = new_grid)
#         left_capacity, new_grid = shiftLine(p1_l, p2_l, max_iter, left = True, expand_along_col = expand_col, grid = new_grid)
        
#         # total_capacity = left_capacity + right_capacity
#         total_capacity = min(left_capacity,right_capacity)*2
#         if total_capacity == 0:
#             total_capacity = 1
#         edge.setCapacity(total_capacity)


def initEdgeDistance(n, edges):
    for edge in edges:     
        dx = n[edge.prev].x - n[edge.next].x
        dy = n[edge.prev].y - n[edge.next].y
        dist = np.linalg.norm(np.array([dx,dy]))
        edge.setDistance(dist)

        if edge.prev == edge.next:
            edge.setProbability(0.1)
        else:
            edge.setProbability(0.5)
    return

def getVoronoiiGraph(occupancy_grid = None, nodes = None, edges = None, start_nodes = None, end_nodes = None):

    # Initialise Edge Distance Attributes
    initEdgeDistance(nodes[:], edges)

    #Initialise Edge Capacity Attributes
    initEdgeCapacity(nodes[:], edges, occupancy_grid)

    G = nx.Graph()

    for e in edges:
        bad_edge = e.edge_attr['distance'] > 20
        if not (bad_edge):
            G.add_edge(e.prev, e.next)
            G.edges[e.prev, e.next]['distance'] = e.edge_attr['distance']
            G.edges[e.prev, e.next]['capacity'] = e.edge_attr['capacity']
            G.edges[e.prev, e.next]['leftCapacity'] = e.edge_attr['leftCapacity']
            G.edges[e.prev, e.next]['rightCapacity'] = e.edge_attr['rightCapacity']
            G.edges[e.prev, e.next]['obs_locations'] = e.edge_attr['obs_locations']
            G.edges[e.prev, e.next]['prev_location'] = e.edge_attr['prev_location']
            G.edges[e.prev, e.next]['next_location'] = e.edge_attr['next_location']
            G.edges[e.prev, e.next]['probability'] = e.edge_attr['probability']
    
    # constrain the capacity of edge with less than certain length
    # for e in G.edges:
    #     if G.edges[e]['distance'] < 1 and G.edges[e]['distance'] > 0:
    #         G.edges[e[0],e[1]]['capacity'] = np.clip(G.edges[e[0],e[1]]['capacity'], 1, 15)

    for n in G.nodes:
        G.nodes[n]['position'] = nodes[n]

    # edges that are overlapping constraint capacity
    # for n in G.nodes:
    #     src = G.nodes[n]['position']
    #     theta_group = {}
    #     theta_e_group = {}
    #     theta_cap = {}
    #     theta_max_cap_edge = {}
    #     for neighbor in G.neighbors(n):
    #         if neighbor == n:
    #             continue
    #         tar = G.nodes[neighbor]['position']
    #         tar_cap = G.edges[n, neighbor]['capacity']
    #         if tar_cap > 0:
    #             x = tar.x - src.x
    #             y = tar.y - src.y
    #             theta = np.arctan2(y,x)*180/np.pi
    #             grp = round(theta/9)*9
    #             if grp in theta_group:
    #                 theta_group[grp] += 1
    #                 theta_e_group[grp].append(neighbor)
    #             else:
    #                 theta_group[grp] = 1
    #                 theta_e_group[grp] = [neighbor]
    #             if (grp in theta_cap and tar_cap >= theta_cap[grp]) or (grp not in theta_cap):
    #                 theta_cap[grp] = tar_cap
    #                 theta_max_cap_edge[grp] = neighbor
        
    #     sorted_theta_group = {k: v for k, v in sorted(theta_group.items(), key=lambda item: item[1], reverse=True)}
    #     for group in sorted_theta_group:
    #         num_elem = sorted_theta_group[group]
    #         if num_elem < 2:
    #             break

    #         to_be_reduced = copy.deepcopy(theta_e_group[group])
    #         to_be_reduced.remove(theta_max_cap_edge[group])
    #         for elem in to_be_reduced:
    #             G.edges[n, elem]['capacity'] = 0

            
    #Set Node capacity
    for n in G.nodes:
        cap = []
        for neighbor in G.neighbors(n):
            cap.append(G.edges[n, neighbor]['capacity'])
        G.edges[n, n]['capacity'] = max(cap)
    G.remove_nodes_from(list(nx.isolates(G)))


    return G