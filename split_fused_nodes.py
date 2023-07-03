from collections import defaultdict, deque
import configargparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.sparse

from sklearn.cluster import SpectralClustering

import graphs


def get_connected_groups(adj_matrix):
    nodes_seen = set()
    groups = []
    queue = deque()
    for node in range(adj_matrix.shape[0]):
        if node in nodes_seen:
            continue
        group = {node}
        nodes_seen.add(node)
        queue.extend( np.where(adj_matrix[node])[0] )
        while queue:
            node2 = queue.pop()
            if node2 in nodes_seen:
                continue
            group.add(node2)
            nodes_seen.add(node2)
            queue.extend( np.where(adj_matrix[node2])[0] )
        groups.append(group)
    return groups


def write_nodes_split(nodes_split, outfile):
    with open(outfile, 'w') as handle:
        handle.write(f'Node_name_before_split\tNode_name1_after_split\tNode_name2_after_split\n')
        for node, (new_name1, new_name2) in nodes_split.items():
            handle.write(f'{node}\t{new_name1}\t{new_name2}\n')


def remove_connections_nodes_split(nodes_split_label1, nodes_split_label2, edges_to_filter):
    t = 0
    for node1 in nodes_split_label1:
        for node2 in nodes_split_label2:
            if edges_to_filter.get(node1, {}).get(node2) is not None:
                t += 1
                edges_to_filter[node1].pop(node2)
    return t


def plot_values(data, label, outfile):
    fig, ax = plt.subplots()
    ax.hist(data, bins=50)
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.set_title(f'{label} counts')
    plt.savefig(outfile)
    plt.close()


def tabulate_dict(d, *extra_dicts, outfile=None, sort=False, sortkey=None, sortreverse=False):
    if outfile:
        handle=open(outfile, 'w')
    s = ''
    if sort:
        itr = sorted(d, key=sortkey, reverse=sortreverse)
    else:
        itr = d
    for key in itr:
        value = d[key]
        line = '{}\t{}'.format(key, value)
        for d2 in extra_dicts:
            line += f'\t{d2.get(key)}'
        line += '\n'
        if outfile:
            handle.write(line)
        s += f'{line}'
    return s


def get_ncuts_sparse(edge_matrix_csc, edge_matrix_csr, index2name, index2name_other_axis):
    cuts = {}
    ncuts = {}
    stored_groups = {}
    # Track how many connected components there are
    group_counts = defaultdict(int)
    
    # Iterate over rows
    for node in range(edge_matrix_csr.shape[0]):
        name = index2name[node]
        slice_node = slice(edge_matrix_csr.indptr[node], edge_matrix_csr.indptr[node+1])
        ids_connected = edge_matrix_csr.indices[slice_node]
        if ids_connected.shape[0] < 4:
            # Minimum edges set to 4
            continue
        connected_connections = edge_matrix_csc[:, ids_connected]
        # Set paths via original node to zero
        connected_connections[node, :] = 0
        connected_connections.eliminate_zeros()
        # Take the two-step adj. matrix
        twostep = connected_connections.T.dot(connected_connections)
        # Convert to dense here (is faster and twostep matrix should be small)
        twostep = np.asarray(twostep.todense())
        # Set self-paths to 0
        np.fill_diagonal(twostep, 0)
        # Get the number of connected components
        connected_components = get_connected_groups(twostep)
        n_conn_components = len(connected_components)
        group_counts[n_conn_components] += 1
        if n_conn_components > 2:
            # Maximum n. of connected components set to 2
            continue
        elif n_conn_components == 2:
            # This should be the best separation: cut will be 0, ncut will be 0 or nan
            # So always cut these. 
            labels = np.fromiter((int(i in connected_components[0]) for i in range(twostep.shape[0])), dtype=np.int64)
        else:
            # spectral graph partitioning
            cluster_fit = cluster_model.fit(twostep)
            labels = cluster_fit.labels_
        res = calc_ncut(labels, twostep, ids_connected)
        if res is not None: # res is None if there are no internal connections within one group and ncut can not be calculated
            cut, ncut, group1, group2 = res
            cuts[name] = cut
            ncuts[name] = ncut
            stored_groups[name] = tuple(tuple(index2name_other_axis[i] for i in group) for group in (group1, group2))
    return cuts, ncuts, stored_groups, group_counts


def get_ncuts(edge_matrix, index2name_axis0, index2name_axis1):
    cuts = {}
    ncuts = {}
    stored_groups = {}
    group_counts = defaultdict(int)

    edge_matrix_csc = edge_matrix
    edge_matrix_csr = edge_matrix.tocsr()
    # Iterate over both axes, i.e., both types of nodes
    for (edge_matrix_columns, edge_matrix_rows,
        index2name_main_axis, index2name_other_axis) in zip((edge_matrix_csc, edge_matrix_csr.T), # Transposing a sparse matrix converts the format between csc and csr
                                                           (edge_matrix_csr, edge_matrix_csc.T),
                                                           (index2name_axis0, index2name_axis1),
                                                           (index2name_axis1, index2name_axis0)):
        cuts_new, ncuts_new, stored_groups_new, group_counts_new = get_ncuts_sparse(edge_matrix_columns, 
                                                                  edge_matrix_rows, index2name_main_axis,
                                                                  index2name_other_axis)
        cuts.update(cuts_new)
        ncuts.update(ncuts_new)
        stored_groups.update(stored_groups_new)
        for n_conn_components, count in group_counts_new.items():
            group_counts[n_conn_components] += count

    return cuts, ncuts, stored_groups, group_counts


def calc_ncut(labels, twostep, ids_connected):
    # Get twostep values between members of different groups
    labels_asym = labels[:, None] ^ labels[None, :]
    cut = int((twostep * labels_asym).sum() / 2)  # Divide by two, or each edge is counted twice 
    twostep_sum = np.asarray(twostep.sum(axis=0)).flatten()
    twostep_total_sum = twostep_sum.sum()
    # Get the sum of edge weights of one group to all other members
    assoc1 = (twostep_sum * labels).sum()
    assoc2 = twostep_total_sum - assoc1
    if assoc1 == 0 or assoc2 == 0:
        return
    # Calculate the normalized cut
    ncut = cut / assoc1 + cut / assoc2
    # Get the group IDS
    mask = np.asarray(labels, bool)
    group1 = ids_connected[mask]
    group2 = ids_connected[~mask]
    return cut, ncut, group1, group2


def apply_cutoffs(ncuts, ncuts_arr, stored_groups, edges_one_way,
                  direct_cutoffs, quantile_cutoffs, name2index_label1, name2index_label2, folder,):
    idmax_label1 = max(int(name.rpartition('_')[2]) for name in name2index_label1)
    idmax_label2 = max(int(name.rpartition('_')[2]) for name in name2index_label2)
    for cutoff_tag, cutoffs in zip(('c', 'q'), (direct_cutoffs, quantile_cutoffs)):
        if cutoffs is None:
            continue
        for cutoff in cutoffs:
            outfolder = os.path.join(folder, f'{cutoff_tag}{cutoff}')
            if not os.path.isdir(outfolder):
                os.mkdir(outfolder)
            if cutoff_tag == 'q':
                if ncuts_arr.size == 0:
                    cutoff = 0
                else:
                    cutoff = np.quantile(ncuts_arr, cutoff)
            current_idmax_label1 = idmax_label1
            current_idmax_label2 = idmax_label2
            edges_new = copy.deepcopy(edges_one_way)
            nodes_to_split = {node for node, ncut in ncuts.items() if ncut < cutoff}
            # split up label1 nodes
            nodes_to_split_label1 = {node for node in nodes_to_split if node.startswith(settings.label1)}
            nodes_to_split_label2 = nodes_to_split - nodes_to_split_label1
            # get connections across the nodes that are going to be split, and remove these 
            # as these could otherwise result in erroneous edges
            n_edges_across_split = remove_connections_nodes_split(nodes_to_split_label1, nodes_to_split_label2, edges_new)

            nodes_split = {}
            for node in nodes_to_split_label1:
                prev_edges = edges_new.pop(node)

                # Give both nodes a new name to prevent confusion when comparing coordinates later
                current_idmax_label1 += 1
                new_name1 = f'{settings.label1}_{current_idmax_label1}'
                current_idmax_label1 += 1
                new_name2 = f'{settings.label1}_{current_idmax_label1}'

                for name, group in zip((new_name1, new_name2), stored_groups.get(node)):
                    for node2 in group:
                        edge_weight = prev_edges.get(node2)
                        if edge_weight is not None: # could be removed if this edge connected two nodes to split
                            edges_new[name][node2] = prev_edges[node2]
                nodes_split[node] = (new_name1, new_name2)
            # split up label2 nodes
            # needs some adjustment if they are connected to nodes of label1 that are also split

            for node in nodes_to_split_label2:
                # Give both nodes a new name to prevent confusion when comparing coordinates later
                current_idmax_label2 += 1
                new_name1 = f'{settings.label2}_{current_idmax_label2}'
                current_idmax_label2 += 1
                new_name2 = f'{settings.label2}_{current_idmax_label2}'

                nodes_split[node] = (new_name1, new_name2)
                # group 1 edges can stay unchanged, but for group2 the node needs to be updated
                for name, group in zip((new_name1, new_name2), stored_groups.get(node)):
                    for node2 in group:
                        edge_weight = edges_new[node2].pop(node, None)
                        if edge_weight is not None:
                            # if edge_weight is None, then this connection no longer exists because of the removal earlier
                            # i.e. this node (of label 1) was itself split
                            edges_new[node2][name] = edge_weight
            outfile = os.path.join(outfolder, 'concatemers_filtered.txt')
            graphs.write_edges(edges_new, outfile)
            summary_file = os.path.join(outfolder, 'correct_doubles_summary.txt')
            with open(summary_file, 'w') as handle:
                handle.write(f'Cutoff\t{cutoff}\n')
                handle.write(f'Nodes_split\t{len(nodes_to_split)}\n')
                handle.write(f'Edges_between_split_nodes\t{n_edges_across_split}\n')

            nodes_split_file = os.path.join(outfolder, 'nodes_split.txt')
            write_nodes_split(nodes_split, nodes_split_file)


def parse_args():
    config_files = [os.path.join(os.path.dirname(__file__), 'minipath.ini')]
    parser = configargparse.ArgParser(default_config_files=config_files,     
                                      description='spurious crosslink filtering script')
    parser.add('-i', '--infile', type=str, required=True, help='The input file')
    parser.add('-o', '--outfolder', type=str, required=True, help='Folder to output the results into')
    parser.add('--label1', default='upi_rg')
    parser.add('--label2', default='upi_by')
    parser.add('--quantiles', nargs='+', type=float, help='The quantiles of normalized cuts below which nodes are split')
    parser.add('--cutoffs', nargs='+', type=float, help='The direct cutoffs of normalized cuts below which nodes are split')

    return parser.parse_args()


if __name__ == '__main__':
    settings = parse_args()
    if not os.path.isdir(settings.outfolder):
        os.mkdir(settings.outfolder)
    # Setup file names
    cuts_file = os.path.join(settings.outfolder, 'cuts.txt')
    ncuts_file = os.path.join(settings.outfolder, 'ncuts.txt')
    logfile = os.path.join(settings.outfolder, 'spectral_node_split_log.txt')
    stored_groups_file = os.path.join(settings.outfolder, 'stored_groups.txt')
    n_conn_components_file = os.path.join(settings.outfolder, 'n_conn_components.txt')

    # setup cluster model for spectral graph partitioning
    cluster_model = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='cluster_qr')

    # Read the pair file
    edges_one_way, names_set_2 = graphs.read_edges_oneway(settings.infile, settings.label1, settings.label2)

    # Reindex all polonies
    name2index_label1, index2name_label1 = graphs.map_umi_ids_to_indeces(edges_one_way)
    name2index_label2, index2name_label2 = graphs.map_umi_ids_to_indeces(names_set_2)
    edges_renamed = graphs.rename_edges_dict(edges_one_way, name2index_label1, name2index_label2)

    # Convert to matrix (sparse, m x n)
    edge_matrix = graphs.convert_edges_to_sparse_matrix_asym_scipy(edges_renamed, names_set_2)

    # Calculate all normalized cuts
    cuts, ncuts, stored_groups, group_counts = get_ncuts(edge_matrix, index2name_label1, index2name_label2)
    ncuts_arr = np.fromiter(ncuts.values(), dtype=np.float64) # Convert to array for easier quantiles

    # Apply cutoffs 
    apply_cutoffs(ncuts, ncuts_arr, stored_groups, edges_one_way,
                  settings.cutoffs, settings.quantiles, name2index_label1, name2index_label2, settings.outfolder)

    # Plot cuts/ncuts
    for label, data in zip(('cuts', 'ncuts'), (cuts, ncuts)):
        plot_values(data.values(), label, os.path.join(settings.outfolder, f'{label}.png'))

    # Log data
    tabulate_dict(cuts, outfile=cuts_file)
    tabulate_dict(ncuts, outfile=ncuts_file)
    tabulate_dict(stored_groups, outfile=stored_groups_file)
    tabulate_dict(group_counts, outfile=n_conn_components_file)

    n_failed_nodes = edge_matrix.shape[0] + edge_matrix.shape[1] - ncuts_arr.shape[0]
    with open(logfile, 'w') as handle:
        handle.write(f'Ncuts_calculated\t{ncuts_arr.shape[0]}\n')
        handle.write(f'Failed_ncuts_calculations\t{n_failed_nodes}\n')
