from core.DataLoader import *
from core.DataStructure import OccupancyGrid
import core.Constant as constant

import numpy as np

class Experiment_Sets:
    def __init__(self, edges = None, nodes = None):
        self.trainable_edges = \
        [(round(nodes[edge.prev].x,2), 
        round(nodes[edge.prev].y,2), 
        round(nodes[edge.next].x,2), 
        round(nodes[edge.next].y,2)) 
        for edge in edges]


    # def optimisable(self, e):
    #     if e in self.edges_to_be_optimised:
    #         return True
    #     return False
    def update_trainable_edges_once(self, edges):
        self.trainable_edges = edges
