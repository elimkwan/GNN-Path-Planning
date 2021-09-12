from core.DataStructure import OccupancyGrid, Point, Edge
from core.AddStartEndNode import addStartEndNode
from core.StandardizeEdgeLength import standardizeEdgeLength
import core.Constant as constant

from scipy.spatial import Voronoi
import numpy as np
import re
import networkx as nx


VOID = -10

def rawInputToArr(scene_map="./input/random-32-32-10/random-32-32-10.map"):
    f = open(scene_map, "r")
    lines = f.readlines()
    f.close()
    
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    img = np.zeros((height,width))
    obstacles = []
    for index in range (4, 4+height):
        cur_line = bytes(lines[index],'utf-8')
        k = np.array(list(cur_line), dtype = np.unicode)
        k = k[:width]
        
        y = np.array(np.where((k == '64')| (k == '84')))
        y = y.reshape((y.shape[1]))
        x = np.tile((index-4), y.shape[0])
        
        pairs = np.stack((x,y), axis = 1)
        obstacles.extend(pairs)
    
    for p in obstacles:
        img[p[0],p[1]] = 1
        
    obstacles = [[x+1.5,y+1.5] for x,y in obstacles] #offset by 1.5 instead of 0.5, will add extra row and column at the begining
    
    #add boundary else Voronoi wont work
    img = np.array(img)
    img = np.insert(img, 0, 1, axis=0)
    img = np.insert(img, img.shape[0], 1, axis=0)
    img = np.insert(img, 0, 1, axis=1)
    img = np.insert(img, img.shape[1], 1, axis=1)
    # print(img.shape)
    
    max_i_row = img.shape[1]-1 #33 #42
    s1 = np.arange(0.5, max_i_row+1, 0.5).round(1)
    s2 = np.tile(0.5, s1.shape[0])
    first_column = np.stack((s1,s2), axis = 1)
    first_column = np.delete(first_column, (0), axis=0)
    first_column = np.delete(first_column, (-1), axis=0)

    #add boundary to obstacle list
    max_i = img.shape[0]-1 #74
    s3 = np.tile(max_i+0.5 , s1.shape[0])
    # last_column = np.stack((s1,s3), axis = 1)
    # last_column = np.delete(last_column, (0), axis=0)
    # last_column = np.delete(last_column, (-1), axis=0)
    s4 = np.tile(max_i_row+0.5 , s1.shape[0])
    last_column = np.stack((s1,s4), axis = 1)

    first_row = np.stack((s2,s1), axis = 1)
    last_row = np.stack((s3,s1), axis = 1)
    
    final_obs = np.concatenate((obstacles, first_column, last_column, first_row, last_row))
    
    return final_obs, img

def rawSceneToArr(
    scene = "./input/random-32-32-10/scen-even/random-32-32-10-even-1.scen", 
    num_agent = 1):

    f = open(scene, "r")
    lines = f.readlines()
    f.close()
    
    starts = []
    ends = []
    pairs = {}
    consider_num_agent = len(lines)-2 # how many additional start end pair
    number_of_agent = consider_num_agent
    c = 0

    # Adjust if there are more maps
    last_char = int(ord(scene.split('/')[-1][:1]))
    # print("Number", last_char)
    if last_char >=49 and last_char <= 53: #check if scene name is the dataset style one
        map_name = 'random-32-32-10'
        if scene.split('/')[-1][2:] == 'den101d.scen':
            map_name = 'den101d'
        elif scene.split('/')[-1][2:] == 'lak109d.scen':
            map_name = 'lak109d'
        elif scene.split('/')[-1][2:] == 'lak105d.scen':
            map_name = 'lak105d'
        elif scene.split('/')[-1][2:] == 'small.scen':
            map_name = 'small'
    else:
        map_name = scene.split('/')[-1][:-9]
    
    for index, l in enumerate(lines):
        x = l.split()
        match = re.match(map_name, x[1])
        if match:
            coord = np.array(x[4:8]).astype('float')
            #+0.5 for centering it; +1 for adding boundary; -1 for starting to count from 0 instead of 1
            # print(coord)
            starts.append([coord[0], coord[1]])
            ends.append([coord[2], coord[3]])
            pairs[Point(coord[0], coord[1])] = Point(coord[2], coord[3])
            
            if c == number_of_agent:
                return starts, ends, pairs
            else:
                c += 1


    print("cant find enough start end pair")
    return starts, ends, pairs

def removeInsidePoint(v, occupancy_grid):
    new_v = []
    removed = []
    for index, p in enumerate (v):
        inside = occupancy_grid.isOccupied(p)
        if (inside):
            removed.append(index)
            new_v.append(Point(VOID, VOID))
            # occupancy_grid.changeOccupiedToFree(p)
            continue

        new_v.append(Point(p[0],p[1]))
    return new_v, removed

def removeInsideLine(length, ridge_vertices, removed_v):
    total_len = length
    new_ridge = []
    for p in ridge_vertices:
        p0 = total_len - abs(p[0]) if p[0] < 0 else p[0]
        p1 = total_len - abs(p[1]) if p[1] < 0 else p[1]
        if p0 in removed_v or p1 in removed_v:
            continue
        new_ridge.append([p0,p1])
    return new_ridge

# def removeOtherInvalidLine(nodes, edges, occupancy_grid):
#     new_edges = []
#     for i, e in enumerate (edges):
#         if occupancy_grid.isValidLine(nodes[e[0]], nodes[e[1]]):
#             new_edges.append(e)
#         # else:
#             # print("removed invalid lines")
#     return new_edges

def removeOtherInvalidLine(nodes, edges, occupancy_grid):
    new_edges = []
    dist_list = []

    for i, e in enumerate (edges):
        if occupancy_grid.isValidLine(nodes[e[0]], nodes[e[1]]):
            p1 = nodes[e[0]]
            p2 = nodes[e[1]]
            dist = np.linalg.norm(np.array([p1.x,p1.y])-np.array([p2.x,p2.y]))
            dist_list.append(dist)
            new_edges.append(e)
        # else:
            # print("removed invalid lines")
    return new_edges, dist_list
           

def cleanNodesEdge(prev_nodes, prev_edges):
    m = {}
    j = 0
    new_nodes = []
    
    for i, node in enumerate(prev_nodes):
        if node == Point(VOID, VOID):
            continue
        m[i] = j #key is previous index, val is desired index
        new_nodes.append(node)
        j += 1
    
    #form list of Edge Object
    new_edges_dir = []
    new_edges = []
    for p, n in prev_edges:
        a = m[p]
        b = m[n]
        a_loc = (new_nodes[a].x, new_nodes[a].y)
        b_loc = (new_nodes[b].x, new_nodes[b].y) 

        new_edges_dir.append(Edge(a, b, a_loc, b_loc))
        new_edges_dir.append(Edge(b, a, b_loc, a_loc))
        new_edges_dir.append(Edge(a, a, a_loc, a_loc))
        new_edges_dir.append(Edge(b, b, b_loc, b_loc))

        new_edges.append(Edge(a, b, a_loc, b_loc))
        new_edges.append(Edge(a, a, a_loc, a_loc))
        new_edges.append(Edge(b, b, b_loc, b_loc))
    
    return new_nodes, new_edges_dir, new_edges

def getNumSubgraphs(edge_list, type='num'):
    G = nx.Graph()
    for e in edge_list:
        if type == 'num':
            G.add_edge(e[0], e[1])
        else:
            G.add_edge(e.prev,e.next)
    gs = [G.subgraph(c) for c in nx.connected_components(G)]
    num_graph = len(gs)
    print("Subgraphs", num_graph)

    gss = sorted(gs, key=len, reverse= True)
    for idx, g in enumerate(gss):
        if len(g.edges) < 2:
            break
    gss = gss[: idx]
    return gss

def getVoronoiiParam(obstacles_loc = None, occupancy_grid = None, start_end_pair = None, num_agent=1):
    # Create Voronoi Diagram
    voronoi = Voronoi(obstacles_loc)

    # Clean Voronoi Diagram
    nodes_tmp, removed_nodes = removeInsidePoint(voronoi.vertices, occupancy_grid)
    edges_tmp = removeInsideLine(len(voronoi.vertices), voronoi.ridge_vertices, removed_nodes)
    edges_tmp, edges_length = removeOtherInvalidLine(nodes_tmp, edges_tmp, occupancy_grid)
    
    # print(edges_tmp)
    if constant.STANDARDIZE_EDGE:
        nodes_tmp, edges_tmp = standardizeEdgeLength(nodes_tmp, edges_tmp, edges_length)

    nodes, edges_dir, edges = cleanNodesEdge(nodes_tmp, edges_tmp)

    # assert getNumSubgraphs(edges_dir, type='edges') == 1
    
    nodes, edges_dir, edges, start_nodes, end_nodes, skipped, occupancy_grid = addStartEndNode(
        occupancy_grid, 
        start_end_pair, 
        nodes, 
        edges_dir, 
        edges, 
        num_agent)

    return nodes, edges_dir, edges, start_nodes, end_nodes, skipped, occupancy_grid


