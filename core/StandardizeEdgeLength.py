from core.DataStructure import Point

from scipy.spatial.distance import cosine
import copy
import numpy as np
import networkx as nx
import pandas as pd

VOID = -10

def canRemoveEdge(org_graph = None, start = None, mid = None, end = None):
    G = copy.deepcopy(org_graph)
    original_num_graph = len([c for c in nx.connected_components(G)])
    for n in mid:
        G.remove_node(n)
    G.add_edge(start, end)
    new_num_graph = len([c for c in nx.connected_components(G)])
    if new_num_graph > original_num_graph:
        return False
    else:
        return True


def popShortEdges(start_node, mid_nodes, end_node, graph):
    # edge_list = copy.deepcopy(e_list)
    # node_list = copy.deepcopy(n_list)
    # removed_edges = []

    canRemove = canRemoveEdge(org_graph = graph, start = start_node, mid = mid_nodes, end = end_node)
    if canRemove:
        for mid_node in mid_nodes:
            graph.remove_node(mid_node)
        graph.add_edge(start_node, end_node)
    # else:
        # print("Cant remove this set of edges", start_node, mid_nodes, end_node)

    return graph

def get_num_neighbors(G, target_node=None, linked_node=None):
    count = 0
    contain_self_edge = False
    suspect = [195, 379]
    for i in G.neighbors(target_node):
        if i == target_node:
            contain_self_edge = True
            continue
        if i != linked_node:
            count += 1

    # if target_node in suspect:
    #     print("Problem ", target_node, count)
    return count



def standardizeEdgeLength(nodes, edges, lengths):
    min_elength = 1

    G = nx.Graph()
    short_edges = []
    long_edges = []
    for idx, e in enumerate(edges):
        G.add_edge(e[0], e[1])
        G.edges[e[0], e[1]]['distance'] = lengths[idx]
        if lengths[idx] < min_elength:
            short_edges.append([e[0], e[1]])
        else:
            long_edges.append([e[0], e[1]])
    for n in G.nodes:
        G.nodes[n]['position'] = nodes[n]


    new_G = copy.deepcopy(G)
    suspects = [91,92,94,93,100]

    # Another attempt to aggregate edges
    for e in short_edges:
        start_node = e[0]
        end_node = e[1]
        mid_nodes = []
        cos_dict = {}
        stopping = False
        loop = 0
        num_edges_formed = 0
        prev_start_end_node = (None, None)

        # print("-----------------------------------------------------------------------------------")
        # print("Before ", len(new_G.edges))

        while not stopping:
            loop += 1
            if prev_start_end_node != None and \
                (prev_start_end_node == (start_node, end_node) or prev_start_end_node == (end_node, start_node)):
                stopping = True
                continue

            num_neighbors = []
            similarities = []
            connected_nodes = []
            target_nodes = []
            

            for cur_node in [start_node, end_node]:
                if cur_node not in new_G.nodes or cur_node in mid_nodes:
                    # print("cur_node", cur_node, "not in graph or is in mid_node")
                    prev_start_end_node = (start_node, end_node)
                    continue
                # if cur_node in suspects:
                #     print("--------------------------------------------")
                #     print("Cur_node: ", cur_node)

                ops_node = end_node if cur_node == start_node else start_node
                a_neighbors = get_num_neighbors(new_G, target_node=cur_node, linked_node=ops_node)
                for n in new_G.neighbors(cur_node):
                    #check how many neighbor this neighbor has
                    b_neighbors = get_num_neighbors(new_G, target_node=n, linked_node=cur_node)
                    neighbors = b_neighbors
                    correct_num_of_neighbors = a_neighbors == 1 and b_neighbors == 1
                    
                    # if n in suspects:
                    #     print("n: ", n)
                    #     print("a_neighbors", a_neighbors, "b_neighbors", b_neighbors)

                    if n == end_node or n == start_node or n in mid_nodes \
                        or not correct_num_of_neighbors:
                        prev_start_end_node = (start_node, end_node)
                        if loop > 5:
                            stopping = True
                        continue
                    num_neighbors.append(neighbors)
                    new_p = new_G.nodes[n]['position']
                    org_p = new_G.nodes[cur_node]['position']
                    similarity = cosine([new_p.x,new_p.y], [org_p.x,org_p.y]) + 1
                    similarities.append(similarity)
                    connected_nodes.append(cur_node)
                    target_nodes.append(n)

            df = pd.DataFrame(list(zip(num_neighbors, similarities, target_nodes, connected_nodes)),
               columns =['Num_neighbors', 'Similarity', 'Target_node', 'Connecting_node'])
            # df.sort_values(['Num_neighbors', 'Similarity'], inplace=True, ascending=[False, False])
            df = df.loc[df['Num_neighbors'] == 1].sort_values(['Similarity'], ascending=[False]).reset_index()


            # expansion_side, target_neighbor = cos_dict[ordered_keys[0]]
            if df.shape[0] < 1:
                prev_start_end_node = (start_node, end_node)
                if loop > 5:
                    stopping = True
                continue

            # print("Expanding start", start_node, "mid_nodes", mid_nodes, "end", end_node)
            num_edges_formed += 1
            _, _, _, target_neighbor, expansion_side = df.iloc[0].tolist()
            if int(expansion_side) == start_node:
                mid_nodes.insert(0, int(expansion_side))
                start_node = int(target_neighbor)
            else:
                mid_nodes.append(int(expansion_side))
                end_node = int(target_neighbor)
            prev_start_end_node = (start_node, end_node)

            p1 = new_G.nodes[start_node]['position']
            p2 = new_G.nodes[end_node]['position']
            dist = np.linalg.norm(np.array([p1.x,p1.y])-np.array([p2.x,p2.y]))

            # if start_node in suspects:
            #     print("start", start_node, "end", end_node, "mid", mid_nodes)

            if dist >= min_elength or loop > 5:
                stopping = True

        if len(mid_nodes) > 0:
            new_G = popShortEdges(
                start_node, 
                mid_nodes, 
                end_node,
                new_G)
            # if start_node in suspects or end_node in suspects:
                # print("Formed Edge - start", start_node, "end", end_node, "mid", mid_nodes)
        # else:
            # if start_node in suspects or end_node in suspects:
                # print("Didnt Form Edge")

        # print("After ", len(new_G.edges))

    new_nodes = copy.deepcopy(nodes)
    new_edges = []
    for e in new_G.edges:
        new_edges.append([e[0], e[1]])
    for i in range (0, len(new_nodes)):
        if i not in new_G.nodes:
            new_nodes[i] = Point(VOID, VOID)



    for e in long_edges:
        if e[0] not in new_G.nodes or e[1] not in new_G.nodes:
            print("Long edge skipped ", e[0], e[1])
            continue

        cur_length = G.edges[e[0], e[1]]['distance']     
        count = 0
        while (cur_length > min_elength):
            num = cur_length/2
            if num < min_elength:
                break
            cur_length = num
            count += 1

        if count != 0:
            p1 = nodes[e[0]]
            p2 = nodes[e[1]]
            points = np.linspace([p1.x,p1.y], [p2.x,p2.y], count+2)
            last_linked_node = e[0]
            for j, p in enumerate(points[1:]):
                new_nodes.append(Point(p[0], p[1]))
                target_node_idx = len(new_nodes)-1
                new_edges.append([last_linked_node, target_node_idx])
                last_linked_node = target_node_idx


    return new_nodes, new_edges
            