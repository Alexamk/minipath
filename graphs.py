# Functions to read and write edges in various formats
# Also some graph algorithms that work on dictionaries of pairs

import numpy as np
import scipy.sparse

from collections import defaultdict, deque


# Dictionary format

def read_edges_oneway(infile, label1, label2):
    reader = get_edge_reader(infile)
    return reader(infile, label1, label2)


def read_edges_short(infile, label1, label2):
    #Return as a dict, and also count the number of names 
    names_by = set()
    edges_one_way = defaultdict(dict)
    with open(infile) as handle:
        for line in handle:
            n1, n2, edge_count = line.strip().split('\t')
            if n1.startswith(label2) and n2.startswith(label1):
                n1, n2 = n2, n1
            count = int(edge_count)
            edges_one_way[n1][n2] = count
            names_by.add(n2)
    return edges_one_way, names_by


def read_edges_long(infile, label1, label2):
    #Return as a dict, and also count the number of names 
    # For the more detailed concatemer files that are generated by the pcr sim
    names_by = set()
    edges_one_way = defaultdict(lambda: defaultdict(int))
    with open(infile) as handle:
        for line in handle:
            n1, n2 = line.strip().split('\t')[0:2]
            if n1.startswith(label2) and n2.startswith(label1):
                n1, n2 = n2, n1
            edges_one_way[n1][n2] += 1
            names_by.add(n2)
    return edges_one_way, names_by


def get_edge_reader(concatemer_file, readers={2: read_edges_short, 3: read_edges_long}):
    # Return the right function, based on whether the file contains 3 entries (short format) or 4 (long format)
    n_tabs = get_ntabs(concatemer_file)
    return readers[n_tabs]


def get_ntabs(concatemer_file):
    with open(concatemer_file) as handle:
        line = handle.readline()
    return line.count('\t')


def write_edges(edges, outfile):
    with open(outfile, 'w') as handle:
        first = True
        for node1, connections in edges.items():
            for node2, edge_weight in connections.items():
                if first:
                    handle.write(f'{node1}\t{node2}\t{edge_weight}')
                    first = False
                else:
                    handle.write(f'\n{node1}\t{node2}\t{edge_weight}')


def count_edges(edges):
    return sum(len(connections) for connections in edges.values())


def count_ueis(edges):
    return sum( sum(connections.values()) for connections in edges.values())


def count_overlap_edges(edges1, edges2):
    # Counts how many edges in the first dict are also in the second dict
    # plus their respective uei counts
    edges_overlap = 0
    ueis_overlap = 0
    for node1, connections in edges1.items():
        connections2 = edges2.get(node1)
        if connections2 is None:
            continue
        for node2, edge_weight in connections.items():
            if node2 in connections2:
                edges_overlap += 1
                ueis_overlap += connections2[node2]
    return edges_overlap, ueis_overlap


def subtract_edges(edges1, edges2):
    # Return a new dict which results from subtracting edges2 from edges1
    # If one edge from edges2 is not in 1, it is ignored
    # otherwise the uei count of the edge in edges1 is decreaed by the uei count in edges2
    # if 0 or lower, it is removed
    # Intended for when edges2 is a subset of edges1
    edges_remaining = defaultdict(lambda: defaultdict(int))
    for node1, connections_e1 in edges1.items():
        connections_e2 = edges2.get(node1)
        if connections_e2 is None:
            edges_remaining[node1] = connections_e1.copy()
            continue
        for node2, edge_weight_e1 in connections_e1.items():
            edge_weight_e2 = connections_e2.get(node2, 0)
            edge_weight_adjusted = edge_weight_e1 - edge_weight_e2
            if edge_weight_adjusted > 0:
                edges_remaining[node1][node2] = edge_weight_adjusted
    return edges_remaining


def fuse_edges(*edge_dicts):
    # Create a new dict with any edge dicts fused together
    new = defaultdict(lambda: defaultdict(int))
    for edge_dict in edge_dicts:
        for node, connections in edge_dict.items():
            for node2, edge_weight in connections.items():
                new[node][node2] += edge_weight
    return new


def iter_ueis(edges):
    # Iterate over all edge weights
    for node1, connections in edges.items():
        for node2, edge_weight in connections.items():
            yield edge_weight


def filter_edges_randomly(edges_dict, sample_fraction):
    filtered_edge_dict = defaultdict(dict)
    names_set2 = set()
    for n1, n2s in edges_dict.items():
        for n2, count in n2s.items():
            if random.random() <= sample_fraction:
                filtered_edge_dict[n1][n2] = count
                names_set2.add(n2)
    return filtered_edge_dict, names_set2


def convert_pair_dict_to_both_ways(edges_one_way, label1, label2):
    edges_both_ways = defaultdict(dict)
    for n1, connections in edges_one_way.items():
        for n2, count in connections.items():
            edges_both_ways[f'{label1}_{n1}'][f'{label2}_{n2}'] = count
            edges_both_ways[f'{label2}_{n2}'][f'{label1}_{n1}'] = count
    return edges_both_ways


def convert_pair_dict_to_single(pair_dict, label1, label2):
    # From both ways pair dict to single
    pair_dict_single = defaultdict(lambda: defaultdict(int))
    nodes_t2 = set()
    for node, pairs in pair_dict.items():
        if node.startswith(label2):
            nodes_t2.add(node)
            continue
        for node2, edge_count in pairs.items():
            pair_dict_single[node][node2] = edge_count
    return pair_dict_single, nodes_t2


def get_connected_components(pairs_dict):
    # From a pairs dictionary, get the connected components
    # Connections must be in there both ways
    groups = []
    queue = deque()
    seen = set()
    for node1 in pairs_dict:
        if node1 in seen:
            continue
        new_group = {node1}
        seen.add(node1)
        queue.extend(pairs_dict[node1])
        while queue:
            node2 = queue.pop()
            if node2 in seen:
                continue
            new_group.add(node2)
            seen.add(node2)
            queue.extend(pairs_dict[node2])
        groups.append(new_group)
    return groups


def get_number_connected_components(pairs_dict):
    # From a pairs dictionary, get the number of connected components
    groups = 0
    queue = deque()
    seen = set()
    for node1 in pairs_dict:
        if node1 in seen:
            continue
        groups += 1
        seen.add(node1)
        queue.extend(pairs_dict[node1])
        while queue:
            node2 = queue.pop()
            if node2 in seen:
                continue
            seen.add(node2)
            queue.extend(pairs_dict[node2])
    return groups

# Matrix format --> converters from edge_dict to matrices

def map_umi_ids_to_indeces(names):
    # Sort the names, then remap them to indeces in order
    # Effectively scale down all identifiers
    # Sort the IDs
    int_ids_sorted = sorted(list(names))
    # Then map
    old2new = {}
    new2old = {}
    for i, id in enumerate(int_ids_sorted):
        old2new[id] = i
        new2old[i] = id
    return old2new, new2old


def rename_edges_dict(edges_dict, name2index_1, name2index_2):
    # Give the edges as a dict, using the indeces of the nodes in the variable list rather than their names 
    edges_out = defaultdict(dict)
    for name1, pair in edges_dict.items():
        for name2, edge_count in pair.items():
            edges_out[name2index_1[name1]][name2index_2[name2]] = edge_count
    return edges_out


def convert_edges_to_sparse_matrix_asym_tf(edges_dict, names_set_2):
    shape = (len(edges_dict), len(names_set_2))
    indices = []
    values = []
    for id1, pairs in edges_dict.items():
        for id2, edge_count in pairs.items():
            indices.append([id1, id2])
            values.append(edge_count)
    values = tf.constant(values, dtype=tf.dtypes.float32)
    return tf.sparse.reorder(tf.sparse.SparseTensor(indices, values, shape))


def convert_edges_to_sparse_matrix_asym_scipy(edges_dict, names_set_2):
    # Of nxm dimensions, where n is the number of polonies of type 1, and m
    # the number of polonies of type 2

    shape = (len(edges_dict), len(names_set_2))
    row_indices = []
    col_indices = []
    values = []

    for id1, pairs in edges_dict.items():
        for id2, edge_count in pairs.items():
            row_indices.append(id1)
            col_indices.append(id2)
            values.append(edge_count)
    return scipy.sparse.csc_matrix((values, (row_indices, col_indices)), shape=shape)


def convert_edges_to_matrix_dense(edges_renamed, names_set_2):
    matrix = np.zeros((len(edges_renamed), len(names_set_2)), dtype=np.int64)
    for id1, pairs in edges_renamed.items():
        for id2, edge_count in pairs.items():
            matrix[id1, id2] = edge_count
    return matrix


def convert_dense_matrix_to_dict(edge_matrix, index2name_axis0, index2name_axis1):
    edges = {}
    xs, ys = np.where(edge_matrix)
    for x, y in zip(xs, ys):
        name1 = index2name_axis0[x]
        name2 = index2name_axis1[y]
        edges[name1, name2] = edge_matrix[x, y]
    return edges


def write_edges_sparse_matrix_asym_scipy(sparse_matrix, index2name_rows, index2name_columns, outfile):
    with open(outfile, 'w') as handle:
        for column_id in range(len(sparse_matrix.indptr)-1):
            column_slice = slice(sparse_matrix.indptr[column_id], sparse_matrix.indptr[column_id+1])
            indices = sparse_matrix.indices[column_slice]
            edge_weights = sparse_matrix.data[column_slice]
            id1 = index2name_columns[column_id]
            for row_id, edge_weight in zip(indices, edge_weights):
                id2 = index2name_rows[row_id]
                handle.write(f'{id1}\t{id2}\t{edge_weight}\n')


