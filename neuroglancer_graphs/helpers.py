
def build_trees(node_ids, locations, edges, node_attrs=None, edge_attrs=None):
    if node_attrs is None:
        node_attrs = {}
    if edge_attrs is None:
        edge_attrs = {}

    node_to_index = {n: i for i, n in enumerate(node_ids)}
    trees = nx.Graph()
    pbs = {
        int(node_id): node_location
        for node_id, node_location in zip(node_ids, locations)
    }
    for i, row in enumerate(edges):
        u = node_to_index.get(int(row[0]), -1)
        v = node_to_index.get(int(row[-1]), -1)

        e_attrs = {attr: values[i] for attr, values in edge_attrs.items()}

        if u == -1 or v == -1:
            continue

        pos_u = daisy.Coordinate(tuple(pbs[node_ids[u]]))
        pos_v = daisy.Coordinate(tuple(pbs[node_ids[v]]))

        if node_ids[u] not in trees.nodes:
            u_attrs = {attr: values[u] for attr, values in node_attrs.items()}
            trees.add_node(node_ids[u], location=pos_u, **u_attrs)
        if node_ids[v] not in trees.nodes:
            v_attrs = {attr: values[v] for attr, values in node_attrs.items()}
            trees.add_node(node_ids[v], location=pos_v, **v_attrs)

        trees.add_edge(node_ids[u], node_ids[v], **e_attrs)

    return trees


def build_trees_from_mst(
    emst, edges_u, edges_v, alpha, coordinate_scale, offset, voxel_size
):
    trees = nx.DiGraph()
    ndims = len(voxel_size)
    for edge, u, v in zip(np.array(emst), np.array(edges_u), np.array(edges_v)):
        if edge[2] > alpha:
            continue
        pos_u = daisy.Coordinate(
            (0,) * (3 - ndims)
            + tuple((u[-ndims:] / coordinate_scale) + (offset / voxel_size))
        )
        pos_v = daisy.Coordinate(
            (0,) * (3 - ndims)
            + tuple((v[-ndims:] / coordinate_scale) + (offset / voxel_size))
        )
        if edge[0] not in trees.nodes:
            trees.add_node(edge[0], location=pos_u)
        else:
            assert trees.nodes[edge[0]]["location"] == pos_u, "locations don't match"
        if edge[1] not in trees.nodes:
            trees.add_node(edge[1], location=pos_v)
        else:
            assert trees.nodes[edge[1]]["location"] == pos_v, "locations don't match"
        trees.add_edge(edge[0], edge[1], d=edge[2])
    return trees