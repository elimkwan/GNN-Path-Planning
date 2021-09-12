import core.Constant as c

import numpy as np
import scipy, scipy.signal

from skimage.transform import resize
from skimage import img_as_bool

class OccupancyGrid:
    def __init__(self, values, origin, resolution):
        original_values = np.array(values)
        self.original_values = original_values
        self.resolution = resolution
        scale = (self.original_values.shape[0]/resolution, self.original_values.shape[1]/resolution)

        grid = np.array(original_values, dtype=np.bool)
        # print("grid", grid)
        transformed = img_as_bool(resize(grid.astype(bool), scale)) #transform to bool to prevent smoothing corners
        transformed = transformed.astype(int)
        # print("transformed", transformed)

        w = int(c.ROBOT_RADIUS * 2/ resolution)
        assert w > 0
        # print("w", w)
        inflat = scipy.signal.convolve2d(transformed , np.ones((w, w)), mode='same')
        # print("inflat", inflat)
        inflat[inflat > 0.] = c.OCCUPIED
        self._values = inflat

    def changeOccupiedToFree(self, position):
        idx = self.get_index(position)
        self._values[idx] = c.FREE


    def get_index(self, position):
        # print("position", position)
        position = np.array(position)
        idx = ((position + self.resolution/2)/self.resolution).astype(np.int32)
        # print("idx", idx)
        # print("self._values.shape[0]", self._values.shape[0])
        idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
        idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
        idx = tuple(idx)
        # print("idx", idx)
        return tuple(idx)

    # def get_position(self, i, j):
    #     return np.array([i, j], dtype=np.float32) * self._resolution + self._origin
    
    def getRows(self):
        return self._values.shape[0]
    
    def getCols(self):
        return self._values.shape[1]
    
    def isOccupied(self, position):
        shape = np.array(self._values).shape
        #shape of (34,34), 33.5 is in bound
        if position[0] >= (shape[0]) or \
            position[1] >= (shape[1]) or \
            position[0] < 0 or \
            position[1] < 0:
            return True
        idx = self.get_index(position)
        return self._values[idx] == c.OCCUPIED

    def isFree(self, p):
        return not self.isOccupied([p.x,p.y])
    
    def isValidLine(self,p1,p2, tolerance = 0):
        d = 20 #descretization 
        clashed = 0
        points = np.linspace([p1.x, p1.y], [p2.x,p2.y], d)
        for p in points:
            if self.isOccupied(p):
                # return False
                clashed += 1
            if clashed > tolerance:
                return False
        return True

    def getArea(self):
        #assume resolution is 1
        return np.count_nonzero(np.array(self.original_values).flatten() == c.FREE)
    
    def set_to_occuiped(self, cur_p1, cur_p2, shifted_p1, shifted_p2, mid_p1, mid_p2):
        # d1 = 4 #descretization along the width
        d2 = 40 #descretization along the height

        line_list = [[cur_p1, cur_p2], [mid_p1, mid_p2], [shifted_p1, shifted_p2]]
        for l in line_list:
            points = np.linspace([l[0].x,l[0].y],[l[1].x,l[1].y], d2)
            shape = np.array(self._values).shape
            for p in points:
                if p[0] >= (shape[0]) or \
                    p[1] >= (shape[1]) or \
                    p[0] < 0 or \
                    p[1] < 0:
                    continue
                idx = self.get_index(p)
                self._values[idx] = c.OCCUPIED

        # idx1 = self.get_index(np.array([shifted_p1.x,shifted_p1.y]))
        # idx2 = self.get_index(np.array([shifted_p2.x,shifted_p2.y]))
        # points = np.linspace([idx1[0], idx1[1]], [idx2[0], idx2[1]], d2)
        # for p in points:
        #     self._values[(p[0],p[1])] = c.OCCUPIED

        return True




class Edge:
    def __init__(self, node_prev, node_next, node_prev_loc, node_next_loc):
        self.prev = node_prev
        self.next = node_next
        self.edge_attr = {
            'distance':0.0, 
            'capacity':0.0, 
            'probability':0.0,

            'leftCapacity':0,
            'rightCapacity':0,

            'prev_location': node_prev_loc,
            'next_location': node_next_loc,
            'obs_locations':[None, None, None, None] 
            #prev-left-obstacle, prev-right-obstacle, next-left-obstacle, next-right-obstacle, 
        }

    def setObsLocations(self, x):
        self.edge_attr['obs_locations'] = x

    def setDistance(self, x):
        self.edge_attr['distance'] = x
        
    def setCapacity(self, l, r):
        self.edge_attr['capacity'] = l+r
        self.edge_attr['leftCapacity'] = l
        self.edge_attr['rightCapacity'] = r
    
    def setProbability(self, x):
        self.edge_attr['probability'] = x


class Point:
    def __init__ (self, x, y):
        self.x = x
        self.y = y