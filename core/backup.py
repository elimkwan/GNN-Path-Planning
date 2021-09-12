def addStartEndNode(occupancy_grid, pairs, nodes, edges_dir, edges, num_agent):
    ZeroDist = 0
    SomeDist = 1
    AdditionalNode = 2
    NotPossible = 3
    
    def appendNode(cur, nodes):
        # check if start end is a impossible node first
        if occupancy_grid.isOccupied([cur.x, cur.y]):
            occupancy_grid.changeOccupiedToFree([cur.x, cur.y])

        distance = np.zeros((len(nodes),2))
        for i, n in enumerate (nodes):
            distance[i] = [np.linalg.norm(np.array([cur.x, cur.y]) - np.array([n.x, n.y])), int(i)]
        
        df = pd.DataFrame(distance)
        df.sort_values(by=0, ascending=True, inplace=True)
        
        for n in df[1]:
            n = int(n)
            dist = df.loc[n, 0]
            if dist == 0.0:
                return ZeroDist, [n]
                # return  SomeDist, n # No matter how close the node is to the current node, add as new node
            if occupancy_grid.isValidLine(cur, nodes[int(n)]):
                return  SomeDist, [n]

        closest = int(df[1][0])
        for i in range (60):
            delta = np.round(2 * np.random.rand(2), 4)
            sign1 = 1 if np.random.rand(1) > 0.5 else 0
            sign2 = 1 if np.random.rand(1) > 0.5 else 0
            mid_node = Point(cur.x + sign1*delta[0], cur.y + sign2*delta[1])
            occuiped = occupancy_grid.isOccupied([mid_node.x, mid_node.y])
            cur_to_middle = occupancy_grid.isValidLine(cur, mid_node)
            middle_to_n = occupancy_grid.isValidLine(mid_node, nodes[closest])
            # print(occuiped, cur_to_middle, middle_to_n)
            if not occuiped and cur_to_middle and middle_to_n:
                return AdditionalNode, [closest, mid_node] 
        # return SomeDist, int(df[1][0]) # add more node to assist
        return NotPossible, None
    
    def appendNode2(case, connected_node, n): # n is the start/end node
        c_n = connected_node[0]

        if case == AdditionalNode:
            print("additional node")
            mid_n = connected_node[1]
            mid_n_idx = len(nodes) + len(new_nodes)
            new_nodes.append(mid_n)
            new_edges_dir.append(Edge(c_n, mid_n_idx))
            new_edges_dir.append(Edge(mid_n_idx, c_n))
            new_edges_dir.append(Edge(c_n, c_n))
            new_edges_dir.append(Edge(mid_n_idx, mid_n_idx))
            new_edges.append(Edge(c_n, mid_n_idx))
            new_edges.append(Edge(c_n, c_n))
            new_edges.append(Edge(mid_n_idx, mid_n_idx))


            startend_n_idx = len(nodes) + len(new_nodes)
            new_nodes.append(n)
            new_edges_dir.append(Edge(mid_n_idx, startend_n_idx))
            new_edges_dir.append(Edge(startend_n_idx, mid_n_idx))
            new_edges_dir.append(Edge(startend_n_idx, startend_n_idx))
            new_edges.append(Edge(startend_n_idx, mid_n_idx))
            new_edges.append(Edge(startend_n_idx, startend_n_idx))
            return startend_n_idx

        
        if case == SomeDist:
            m = len(nodes) + len(new_nodes)
            new_nodes.append(n)
            new_edges_dir.append(Edge(c_n, m))
            new_edges_dir.append(Edge(m, c_n))
            new_edges_dir.append(Edge(c_n, c_n))
            new_edges_dir.append(Edge(m, m))

            new_edges.append(Edge(c_n, m))
            new_edges.append(Edge(c_n, c_n))
            new_edges.append(Edge(m, m))
            return m
        elif case == ZeroDist:
            return c_n
        return None
    
    new_nodes, new_edges_dir, new_edges = [], [], []
    start_nodes, end_nodes = [], []
    count = 0
    skipped = []
    for idx, start in enumerate(pairs):
        case1, connected_node1 = appendNode(start, nodes)
        case2, connected_node2 = appendNode(pairs[start], nodes)
        # print("index", idx, case1, case2, connected_node1, connected_node2)
        
        if (case1 == NotPossible or case2 == NotPossible):
            # print("Impossible", idx, case1, case2)
            skipped.append(idx)
            continue
        
        print("Case", idx, case1, case2)
        count += 1
        mm = appendNode2(case1, connected_node1, start)
        if mm != None:
            start_nodes.append(mm)
        mm = appendNode2(case2, connected_node2, pairs[start])
        if mm != None:
            end_nodes.append(mm)
        # start_nodes.append(connected_node1)
        # end_nodes.append(connected_node2)

        if count == num_agent:
            break
    
    for k in new_nodes:
        print("new nodes", k.x, k.y)

    print(len(nodes))
    print(len(new_nodes))

    nodes.extend(new_nodes)   
    edges_dir.extend(new_edges_dir)
    edges.extend(new_edges)

    return nodes, edges_dir, edges, start_nodes, end_nodes, skipped, occupancy_grid