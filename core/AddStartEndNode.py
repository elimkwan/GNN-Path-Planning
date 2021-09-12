from core.DataStructure import OccupancyGrid, Point, Edge
import pandas as pd
import numpy as np

ZeroDist = 0
SomeDist = 1
AdditionalNode = 2
NotPossible = 3
Infinity = 1000

def getNeighbors(cur, nodes, occupancy_grid):
    """
    return Case Number, List of Connecting Nodes 
    (Connecting Nodes = nodes that will be used to add the start end location to the graph)
    """

    # Sort all nodes by distance w.r.t to the current target node
    distance = np.zeros((len(nodes),2))
    for i, n in enumerate (nodes):
        if n.x < 0 or n.y < 0: # Invalid points dont get chosen as start end
            # print("Invalid points dont get chosen as start end")
            dist = Infinity
        else:
            dist = np.linalg.norm(np.array([cur.x, cur.y]) - np.array([n.x, n.y]))
        distance[i] = [dist, int(i)]
    df = pd.DataFrame(distance)
    df.sort_values(by=0, ascending=True, inplace=True)
    df = df.loc[df[0] < Infinity]
    ordered_nodes = df[1]
    # print("ordered_nodes ", df)

    for n in ordered_nodes:
        n = int(n)
        dist = df.loc[n, 0]
        # print("n", n, "dist", dist)
        if dist == 0.0:
            # print("ZeroDist")
            return ZeroDist, [n]
        if occupancy_grid.isValidLine(cur, nodes[int(n)]):
            # print("SomeDist")
            return  SomeDist, [n]


    # Add single hop if direct valid line couldnt be found
    if len(ordered_nodes) > 0:
        # print(ordered_nodes)
        closest = int(ordered_nodes.iloc[0])
        for i in range (60):
            mid_node = sampleValidPoint(occupancy_grid, cur)
            if mid_node == None:
                continue

            cur_to_middle = occupancy_grid.isValidLine(cur, mid_node)
            middle_to_n = occupancy_grid.isValidLine(mid_node, nodes[closest])
            if cur_to_middle and middle_to_n:
                return AdditionalNode, [closest, mid_node]

    # Return impossible
    return NotPossible, None

def sampleValidPoint(occupancy_grid, cur):
    for i in range (100):
        delta = np.round(2 * np.random.rand(2), 4)
        sign1 = 1 if np.random.rand(1) > 0.5 else -1
        sign2 = 1 if np.random.rand(1) > 0.5 else -1
        mid_node = Point(cur.x + sign1*delta[0], cur.y + sign2*delta[1])
        occuiped = occupancy_grid.isOccupied([mid_node.x, mid_node.y])
        if not occuiped:
            return mid_node
    return None


class newGraphAttributes():
    def __init__(self, num_original_nodes, original_nodes):
        self.original_nodes = original_nodes
        self.base = num_original_nodes
        self.start_nodes = []
        self.end_nodes = []
        self.new_nodes = []
        self.new_edges = []
        self.new_edges_dir = []

    def appendNode(self, node_val):
        self.new_nodes.append(node_val) 

    def appendEdge(self, n1_id, n2_id, n1, n2):
        n1_loc = (n1.x, n1.y)
        n2_loc = (n2.x, n2.y)
        self.new_edges_dir.append(Edge(n1_id, n2_id, n1_loc, n2_loc))
        self.new_edges_dir.append(Edge(n2_id, n1_id, n2_loc, n1_loc))
        self.new_edges_dir.append(Edge(n1_id, n1_id, n1_loc, n1_loc))
        self.new_edges_dir.append(Edge(n2_id, n2_id, n2_loc, n2_loc))
        self.new_edges.append(Edge(n1_id, n2_id, n1_loc, n2_loc))
        self.new_edges.append(Edge(n1_id, n1_id, n1_loc, n1_loc))
        self.new_edges.append(Edge(n2_id, n2_id, n2_loc, n2_loc))

    def add(self, case, connected_node, target_node, isStartNode = True):
        new_node_id = self.base + len(self.new_nodes)
        new_node_loc = target_node # Node object
        closest_neighbor = connected_node[0]
        closest_neighbor_loc = self.original_nodes[closest_neighbor]
        # print("closest_neighbor", closest_neighbor)
        # print("case", case)

        if case == SomeDist or case == ZeroDist:
            self.appendNode(new_node_loc)
            self.appendEdge(new_node_id, closest_neighbor, new_node_loc, closest_neighbor_loc)


        if case == AdditionalNode:
            mid_node_id = new_node_id + 1
            mid_node_loc = connected_node[1]

            self.appendNode(new_node_loc)
            self.appendNode(mid_node_loc)
            self.appendEdge(new_node_id, mid_node_id, new_node_loc, mid_node_loc)
            self.appendEdge(mid_node_id, closest_neighbor, mid_node_loc, closest_neighbor_loc)
            # print("mid node")

        # if case == ZeroDist:
            # Do Nothing

        # print("new_node_id", new_node_id)

        if isStartNode:
            self.start_nodes.append(new_node_id)
        else:
            self.end_nodes.append(new_node_id)

def addStartEndNode(occupancy_grid, pairs, nodes, edges_dir, edges, num_agent):
    # new_nodes, new_edges_dir, new_edges = [], [], []
    # start_nodes, end_nodes = [], []
    count = 0
    skipped = []

    newGraphAttribute = newGraphAttributes(len(nodes), nodes)

    for idx, start in enumerate(pairs):
        cases = [NotPossible, NotPossible]
        connected_nodes = [None,None]
        
        for k, target in enumerate([start, pairs[start]]):
            # check if start end is a impossible node first
            if occupancy_grid.isOccupied([target.x, target.y]):
                occupancy_grid.changeOccupiedToFree([target.x, target.y])
                # cases[k] = NotPossible
                # connected_nodes[k] = None
            else:
                cases[k], connected_nodes[k] = getNeighbors(target, nodes, occupancy_grid)

        # If any of the start or end node is impossible to embed into the graph, discard this pair of start-end node
        if cases[0] == NotPossible or cases[1] == NotPossible :
            # print("Not Possible to add", str(target.x), str(target.y))
            skipped.append(idx)
            continue

        for k, target in enumerate([start, pairs[start]]):
            isStartNode = True if k ==0 else False
            newGraphAttribute.add(cases[k], connected_nodes[k], target, isStartNode)

        count += 1 # Count how many pairs added as number of agent added
        if count == num_agent:
            break

    # print("len nodes", len(nodes))
    # print("len newGraphAttribute.new_nodes", len(newGraphAttribute.new_nodes))

    nodes.extend(newGraphAttribute.new_nodes)   
    edges_dir.extend(newGraphAttribute.new_edges_dir)
    edges.extend(newGraphAttribute.new_edges)

    return nodes, edges_dir, edges, newGraphAttribute.start_nodes, newGraphAttribute.end_nodes, skipped, occupancy_grid





# def appendNode(
#     case, 
#     connected_node, 
#     target_node): # target_node is the start/end node
#     c_n = connected_node[0]

#     if case == AdditionalNode:
#         print("additional node")
#         mid_n = connected_node[1]
#         mid_n_idx = len(nodes) + len(new_nodes)
#         new_nodes.append(mid_n)
#         new_edges_dir.append(Edge(c_n, mid_n_idx))
#         new_edges_dir.append(Edge(mid_n_idx, c_n))
#         new_edges_dir.append(Edge(c_n, c_n))
#         new_edges_dir.append(Edge(mid_n_idx, mid_n_idx))
#         new_edges.append(Edge(c_n, mid_n_idx))
#         new_edges.append(Edge(c_n, c_n))
#         new_edges.append(Edge(mid_n_idx, mid_n_idx))


#         startend_n_idx = len(nodes) + len(new_nodes)
#         new_nodes.append(n)
#         new_edges_dir.append(Edge(mid_n_idx, startend_n_idx))
#         new_edges_dir.append(Edge(startend_n_idx, mid_n_idx))
#         new_edges_dir.append(Edge(startend_n_idx, startend_n_idx))
#         new_edges.append(Edge(startend_n_idx, mid_n_idx))
#         new_edges.append(Edge(startend_n_idx, startend_n_idx))
#         return startend_n_idx

    
#     if case == SomeDist:
#         m = len(nodes) + len(new_nodes)
#         new_nodes.append(n)
#         new_edges_dir.append(Edge(c_n, m))
#         new_edges_dir.append(Edge(m, c_n))
#         new_edges_dir.append(Edge(c_n, c_n))
#         new_edges_dir.append(Edge(m, m))

#         new_edges.append(Edge(c_n, m))
#         new_edges.append(Edge(c_n, c_n))
#         new_edges.append(Edge(m, m))
#         return m
#     elif case == ZeroDist:
#         return c_n
#     return None