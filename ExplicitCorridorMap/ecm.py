import numpy as np
# import os
import multiprocessing
from shapely.geometry import LineString
from core.DataStructure import OccupancyGrid


LEFT = 0
RIGHT = 1
OUT = 2

class EdgeOrientation:
    def __init__(self, v1, v2):
        a = v1
        b = v2
        cd_length = 6

        ab = LineString([a, b])
        left = ab.parallel_offset(cd_length / 2, 'left')
        right = ab.parallel_offset(cd_length / 2, 'right')

        # Info for b
        mid = b
        c = left.boundary[1]
        d = right.boundary[0]  # note the different orientation for right offset
        cd = LineString([c, d])
        lbound = np.arctan2(c.x-mid[0], c.y-mid[1]) * 180/np.pi
        rbound = np.arctan2(d.x-mid[0], d.y-mid[1]) * 180/np.pi
        if rbound < 0: # Exchange Left Right, Left bound is always negative
            exchange = rbound 
            rbound = lbound
            lbound = exchange
        # print("Left ", lbound, "Right ", rbound)
        
        self.c = c
        self.d = d
        self.rbound = rbound
        self.lbound = lbound
        
    def getOrientation(self, angle):
        # Ensure angle is in 0-360 spectrum
        if angle >= self.rbound and angle <= (self.rbound+90):
            return RIGHT
        elif angle > (self.rbound+90) and angle <= (self.rbound+180):
            return LEFT
        else:
            return OUT

class ECM:
    def __init__(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid

    def getClosestObstaclePt(self, edge):
        if edge[0][0] == edge[1][0] and edge[0][1] == edge[1][1]:
            return [None, None, None, None] # Single vertex line

        resolution = 5
        obs = [{},{}]
        bending_points = [] #v1-left, v1-right, v2-left, v2-right,
        top_vertex = edge[0] if edge[0][1] > edge[1][1] else edge[1]
        bottom_vertex = edge[1] if edge[0][1] > edge[1][1] else edge[0]
        line = EdgeOrientation(bottom_vertex, top_vertex)
        for i, center in enumerate(edge): 
            for ray in range (0, 360+1, resolution):
                orientation = line.getOrientation(ray)
                if orientation == OUT:
                    continue
                encountered, coord = self.findIntersection(center, ray * np.pi / 180, self.occupancy_grid)
                if encountered:
                    r = coord[-1]
                    if r in obs[orientation].keys():
                        obs[orientation][r].append(coord[0:2])
                    else:
                        obs[orientation][r] = [coord[0:2]]
            for orientation in ([LEFT, RIGHT]):
                radius = sorted(obs[orientation].keys())
                if len(radius) > 0:
                    temp = obs[orientation][radius[0]][0] # radius[0] for closest obs
                    # perpendicular = [[line.c.x, line.c.y], [line.d.x, line.d.y]]
                    # obs_line = [[temp[0], temp[1]], [0,0]]
                    # projection = np.dot(perpendicular, obs_line)
                    obstacles_at_r = (temp[0], temp[1], radius[0])
                else:
                    print("Cant find any obstacle")
                    obstacles_at_r = (0,0,0)
                bending_points.append(obstacles_at_r)
        return bending_points

    # def isOccuiped(pt):
    #     # dummy function
    #     if pt[0] > 1 and pt[1] > 1:
    #         return True
    #     return False

    def findIntersection(self, center, angle, occupancy_grid):
        searchRange =  np.linspace(0.1, 30, 60)
        searchRange = np.around(searchRange, 4)
        last_pt = center
        for r in searchRange:
            x = round(center[0] + r * np.sin(angle),2)
            y = round(center[1] + r * np.cos(angle),2)
            occupied = occupancy_grid.isOccupied((x,y))
            if occupied:
                return True, (last_pt[0],last_pt[1],r)
            last_pt = (x, y)
        return False, None


def explicitCorridorMap(edges, occupancy_grid):
    # edges are in the format of:
    # edges = [
    #     [(0,0), (5,5)]
    # ]

    ecm_obj = ECM(occupancy_grid)
    # os.system("taskset -p 0xff %d" % os.getpid())
    pool_obj = multiprocessing.Pool()
    bending_points = pool_obj.map(ecm_obj.getClosestObstaclePt, edges)
    return bending_points