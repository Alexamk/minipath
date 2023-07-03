from collections import defaultdict
import configargparse
import itertools
import logging
import matplotlib.pyplot as plt
import numba
import numpy as np
import os
import scipy.sparse
import time

import graphs

# Indirect paths at three steps for bipartite graphs can be calculated
# either by going taking one step from type 1 to others of type 1, excluding self connections,
# and then taking one more step from there to type 2 (only if connected in the original)
# or the reverse (type 2 to type 2, then one more step to type 1)

@numba.njit(fastmath=True, parallel=True)
def calc_indirect_paths_connected_s2v1(spairs_indptr, spairs_indices, spairs_data, 
                                    spairs_squared_t1_indptr, spairs_squared_t1_indices, spairs_squared_t1_data,
                                    ):
    # For asymmetric adjacency matrices of bipartite graphs
    # With numba acceleration
    new_data = np.empty_like(spairs_data, dtype=spairs_data.dtype)
    for column_id in numba.prange(len(spairs_indptr) -1): # node 1
        slice_spairs = slice(spairs_indptr[column_id], spairs_indptr[column_id+1])
        indices_spairs = spairs_indices[slice_spairs]
        edge_weights_spairs = spairs_data[slice_spairs]
        length_column = len(indices_spairs) # n. of values
        for j in numba.prange(length_column): # node 2
            # Get the corresponding column from spairs_squared
            # should actually be row but since it is a symmetrical matrix it doesn't matter
            # and getting a column is easier since it is a csc matrix
            row_id = indices_spairs[j]
            connection_n1_n2 = edge_weights_spairs[j]
            slice_spairssq = slice(spairs_squared_t1_indptr[row_id], spairs_squared_t1_indptr[row_id+1])
            indices_spairssq = spairs_squared_t1_indices[slice_spairssq]
            edge_weights_spairssq = spairs_squared_t1_data[slice_spairssq]
            # Find overlap by mergesorting (adapated from numpy's intersect1d code)
            aux = np.concatenate((indices_spairs, indices_spairssq))
            aux_sort_indices = np.argsort(aux, kind='mergesort')
            new_weight = 0
            # Look at every node3 that is two steps from node 2 and one step from node 1
            # if both node 1 and node 2 are connected (directly and with two steps, respectively)
            # this means that there is an indirect path
            for k in range(len(aux_sort_indices) - 1): # node 3
                i1 = aux_sort_indices[k]
                i2 = aux_sort_indices[k+1]
                if aux[i1] == aux[i2]: # I.e., node 1 is connected in spairs and node 2 in spairs_squared_t1
                    connection_n1_n3 = edge_weights_spairs[i1]
                    # Adjust n2-n3 indirect paths to remove any path that went through n1
                    connection_n2_n3 = edge_weights_spairssq[i2-length_column] - (connection_n1_n2 * connection_n1_n3)
                    new_weight += ( connection_n1_n3 * connection_n2_n3 )
            new_data[spairs_indptr[column_id] + j] = new_weight
    return new_data


def minpath_dense(np_pairs):
    # For asymmetric adjacency matrices of bipartite graphs
    pairs_cubed = np_pairs @ np_pairs.T @ np_pairs
    # The number of edges squared represents, for each node, the sum of all its edges squared
    edges_sq = np_pairs**2
    nedges_sq_axis0 = np.asarray(edges_sq.sum(axis=0))
    nedges_sq_axis1 = np.asarray(edges_sq.sum(axis=1))
    # Expected values for direct links between two nodes after taking three steps
    # are then found by adding these together for two connected nodes,
    # subtracting the edge squared itself (otherwise it would be counted double)
    # and multiplying by their respective edge
    direct_links = (nedges_sq_axis0[None, :] + nedges_sq_axis1[:, None] - edges_sq) * np_pairs
    indirect_paths = pairs_cubed - direct_links
    ip = np.where(np_pairs, indirect_paths, 0)
    return ip


def sparse_wrapper(spairs, spairs_squared_t1):
    values = calc_indirect_paths_connected_s2v1(spairs.indptr, spairs.indices, spairs.data, 
                               spairs_squared_t1.indptr, spairs_squared_t1.indices, spairs_squared_t1.data,
                               )
    matrix = scipy.sparse.csc_matrix((values, np.copy(spairs.indices), np.copy(spairs.indptr)), shape=spairs.shape)
    return matrix


def filter_and_convert(spairs, indirect_paths, cutoff, index2name_rows, index2name_columns):
    edges_out = defaultdict(lambda: defaultdict(int))
    for column_id in range(len(spairs.indptr)-1):
        column_slice = slice(spairs.indptr[column_id], spairs.indptr[column_id+1])
        indices = spairs.indices[column_slice]
        edge_weights = spairs.data[column_slice]
        indirect_paths_slice = indirect_paths.data[column_slice]
        id1 = index2name_columns[column_id]
        for row_id, edge_weight, indirect_path in zip(indices, edge_weights, indirect_paths_slice):
            if indirect_path >= cutoff:
                id2 = index2name_rows[row_id]
                edges_out[id2][id1] = edge_weight
    return edges_out


def filter_and_write(spairs, indirect_paths, cutoff, outfile, index2name_rows, index2name_columns):
    with open(outfile, 'w') as handle:
        for column_id in range(len(spairs.indptr)-1):
            column_slice = slice(spairs.indptr[column_id], spairs.indptr[column_id+1])
            indices = spairs.indices[column_slice]
            edge_weights = spairs.data[column_slice]
            indirect_paths_slice = indirect_paths.data[column_slice]
            id1 = index2name_columns[column_id]
            for row_id, edge_weight, indirect_path in zip(indices, edge_weights, indirect_paths_slice):
                if indirect_path >= cutoff:
                    id2 = index2name_rows[row_id]
                    handle.write(f'{id1}\t{id2}\t{edge_weight}\n')


def filter_and_write_multiple(spairs, indirect_paths, outfiles, index2name_rows, index2name_columns):
    for column_id in range(len(spairs.indptr)-1):
        column_slice = slice(spairs.indptr[column_id], spairs.indptr[column_id+1])
        indices = spairs.indices[column_slice]
        edge_weights = spairs.data[column_slice]
        indirect_paths_slice = indirect_paths.data[column_slice]
        id1 = index2name_columns[column_id]
        for row_id, edge_weight, indirect_path in zip(indices, edge_weights, indirect_paths_slice):
            id2 = index2name_rows[row_id]
            for cutoff, handle in outfiles.items():
                if indirect_path >= cutoff:
                    handle.write(f'{id1}\t{id2}\t{edge_weight}\n')


def convert_sparse_edges_to_edge_dict(spairs, index2name_rows, index2name_columns):
    edge_dict = defaultdict(dict)
    for column_id in range(len(spairs.indptr)-1):
        column_slice = slice(spairs.indptr[column_id], spairs.indptr[column_id+1])
        indices = spairs.indices[column_slice]
        edge_weights = spairs.data[column_slice]
        id1 = index2name_columns[column_id]
        for row_id, edge_weight in zip(indices, edge_weights):
            id2 = index2name_rows[row_id]
            edge_dict[id1][id2] = edge_weight
    return edge_dict


def plot_hist(indirect_paths, logbins, logy, folder):
    basefile = 'indirect_paths'
    if logbins:
        basefile += '_symlogx'
    if logy:
        basefile += '_logy'
    plotfile = os.path.join(folder, f'{basefile}.png')
    fig, ax = plt.subplots()
    bins=50
    if logbins:
        max_path = max(indirect_paths.data)
        if max_path > 0:
            max_bin = int(np.ceil(np.log10(max_path)))
            bins = np.logspace(0, max_bin, 50)
            # insert extra bin to see indirect paths of 0
            # not really a log distribution anymore, but can't see these otherwise
            bins = np.insert(bins, 0, 0)
    ax.hist(indirect_paths.data, bins=bins)
    if logy:
        ax.set_yscale('log')
    if logbins:
        ax.set_xscale('symlog')
    ax.set_xlabel('Indirect path')
    ax.set_ylabel('Number of edges')
    ax.set_title('Indirect path count distribution per edge')
    fig.tight_layout()
    plt.savefig(plotfile)
    plt.close(fig)


def parse_args():
    config_files = [os.path.join(os.path.dirname(__file__), 'minipath.ini')]
    parser = configargparse.ArgParser(default_config_files=config_files,     
                                      description='spurious crosslink filtering script')
    parser.add('-i', '--infile', type=str, help='The file containing the pairs', required=True)
    parser.add('-o', '--outfolder', type=str, help='The folder in which the results will be put', required=True)
    parser.add('--label1', type=str, help='The label of nodes type 1')
    parser.add('--label2', type=str, help='The label of nodes type 2')
    parser.add('--quantiles', nargs='+', type=float, help='The quantiles of lowest indirect paths to remove')
    parser.add('--cutoffs', nargs='+', type=float, help='The direct cutoffs to apply')
    parser.add('--mode', choices=['sparse', 'dense'], default='sparse')
    return parser.parse_args()


if __name__ == '__main__':
    settings = parse_args()
    if not os.path.isdir(settings.outfolder):
        os.mkdir(settings.outfolder)
    # Read a pair file
    edges_one_way, names_set_2 = graphs.read_edges_oneway(settings.infile, settings.label1, settings.label2)
    # Reindex all polonies
    name2index_rows, index2name_rows = graphs.map_umi_ids_to_indeces(edges_one_way)
    name2index_columns, index2name_columns = graphs.map_umi_ids_to_indeces(names_set_2)
    edges_renamed = graphs.rename_edges_dict(edges_one_way, name2index_rows, name2index_columns)
    # Convert to sparse matrix (not full NxN but one type as rows and the other as columns)
    print('Converting to matrix')
    spairs = graphs.convert_edges_to_sparse_matrix_asym_scipy(edges_renamed, names_set_2)

    if settings.mode == 'dense':
        print('Converting sparse pairs to dense pairs')
        t0 = time.time()
        pairs_dense = np.asarray(spairs.todense())
        t1 = time.time()
        print(f'Time taken: {t1-t0}')
        print('Calculating indirect paths')
        t0 = time.time()
        indirect_paths_connected = minpath_dense(pairs_dense)
        t1 = time.time()
        print(f'Time taken: {t1-t0}')

    elif settings.mode == 'sparse':
        print('Calculating spairs_squared')
        t0 = time.time()
        spairs_squared_t1 = spairs @ spairs.T
        spairs_squared_t1.setdiag(0)
        spairs_squared_t1.eliminate_zeros()
        t1 = time.time()
        print(f'Time taken: {t1-t0}')
        print('Calculating indirect paths')
        t0 = time.time()
        indirect_paths_connected = sparse_wrapper(spairs, spairs_squared_t1)
        t1 = time.time()
        print(f'Time taken: {t1-t0}')

    ipfile = os.path.join(settings.outfolder, 'indirect_paths_connected.txt')
    graphs.write_edges_sparse_matrix_asym_scipy(indirect_paths_connected, index2name_rows, index2name_columns, ipfile)

    # Plot a histogram of the values
    for logbins, logy in itertools.product((True, False), repeat=2):
        plot_hist(indirect_paths_connected, logbins, logy, settings.outfolder)

    # Set cutoffs for quantiles of edges
    outfiles = {}
    if settings.quantiles:
        quantiles = settings.quantiles
        cutoffs = np.quantile(indirect_paths_connected.data, quantiles)
        with open(os.path.join(settings.outfolder, 'quantiles.txt'), 'w') as handle:
            handle.write(f'{quantiles}\n{list(cutoffs)}')
        for cutoff, q in zip(cutoffs, quantiles):
            if cutoff > 0:
                filtered_pairs_file = os.path.join(settings.outfolder, f'minpath_filtered_pairs_q{q}.txt')
                outfiles[cutoff] = open(filtered_pairs_file, 'w')
    if settings.cutoffs:
        for cutoff in settings.cutoffs:
            if cutoff > 0:
                filtered_pairs_file = os.path.join(settings.outfolder, f'minpath_filtered_pairs_c{cutoff}.txt')
                outfiles[cutoff] = open(filtered_pairs_file, 'w')
    if outfiles:
        filter_and_write_multiple(spairs, indirect_paths_connected, outfiles, index2name_rows, index2name_columns)
        for handle in outfiles.values():
            handle.close()
