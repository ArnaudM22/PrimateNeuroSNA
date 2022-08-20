""" This module contains tools for the analysis of cerebral and social networks.

It contains :
1. The "indiv_properties" function allowing the calculation of properties at 
the level of individuals and dyads (eigenvector centrality, brokerage and distance). 
2. The "net_properties" function allowing the calculation of properties at 
the scale of the whole network (clustering and local efficiency).
3. The "comp_network_properties" function allowing the comparison of networks 
between them (clustering and local efficiency differences, and deltacon if known node correspondence). 
4. The "reciprocity" function to perform a reciprocity analysis for oriented networks. 
5. The "postnet_NM" function allowing to  build a list of adjacency matrices 
of random networks according to the post-network Null Model.

NB: There is also the "plot_net_properties", an helper function providing plots 
for "net_properties" and "comp_network_properties".

See the docstrings and the report for a detailed description.

"""

import pandas as pd
import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import random
__author__ = " Arnaud Maupas "
__contact__ = "arnaud.maupas@ens.fr"
__date__ = "19/09/22"
__version__ = "1"


def indiv_properties(data):
    """Computes properties at the level of individuals and dyads (eigenvector centrality, brokerage and distance).

    Parameters
    ----------
    data : dataframe
        Adjacency matrix of the network of interest.

    Returns
    -------
    centrality_brokerage : dataframe
        Eigenvector centrality and brokerage value for each node.
    distance_dict : dict of dict
        Distance between nodes.
    """
    # The adjacency matrix is onverted to a networkx object.
    net = nx.Graph(data)
    # Computation of centrality and brokerage.
    centrality_brokerage = pd.Series(nx.eigenvector_centrality(net)).rename(
        'centrality').to_frame().join((1/pd.Series(nx.constraint(net))).rename('brokerage'))
    # Distances computation.
    distance_dict = dict(nx.all_pairs_dijkstra_path_length(net))

    return centrality_brokerage, distance_dict


def postnet_NM(data, seed=None, nb_iter=1000):
    """Builds a list of adjacency matrices of random networks according to the pre-network Null Model.

    This function works in 3 steps: 
    1. First, it produces random edge weights list. These lists are obtained by 
    performing a bootstrap random draw from the edge weights list of the basic network. 
    We produce as many lists of weights of random edges as there are random 
    networks (number defined by nb_iter).
    2. We produce in parallel a list of random networks following the Erdős-Rényi model.
    These networks have the same number of edges and nodes as our basic network.
    3. We associate each list of weights of the random edges with each random network, 
    and we define the name of the nodes of the random networks according to the nodes of the original network.

    Parameters
    ----------
    data : dataframe
        Adjacency matrix of the network of interest.
    seed : int, optional
        Seed for random draws. The default is None.
    nb_iter : int, optional
        Number of random networks to build. The default is 1000.

    Returns
    -------
    adj_list : list of dataframes
        List of random graphs adjacency matrix.
    """

    # The seed is defined (optional).
    if seed:
        random.seed(seed)
    # The adjacency matrix is onverted to a networkx object.
    net = nx.Graph(data)
    # The nodes names are retrieved.
    nodes_names = list(net.nodes)
    # The weight list is retrieved.
    weight_list = list(nx.get_edge_attributes(net, 'weight').values())
    # The number of edges and nodes are retrieved.
    n_nodes = len(nodes_names)
    n_edges = len(weight_list)
    # We practice a bootstrap sampling on the weight list.
    resampled_weight_list = [None] * nb_iter
    resampled_weight_list = list(map(lambda x: random.choices(
        weight_list, k=n_edges), resampled_weight_list))
    # List of Erdős-Rényi graphs.
    adj_list = []
    for i in range(nb_iter):
        adj_list.append(nx.gnm_random_graph(n_nodes, n_edges))
    # Weights setting.
    set_weight_list = list(map(lambda x, y: dict(
        zip(x.edges, y)), adj_list, resampled_weight_list))
    for x, y in zip(adj_list, set_weight_list):
        nx.set_edge_attributes(x, y, 'weight')
    # A dict to set the node names is created.
    raname_dict = {k: nodes_names[k] for k in range(len(nodes_names))}
    adj_list = list(map(lambda x: nx.relabel_nodes(x, raname_dict), adj_list))
    adj_list = list(map(lambda x: nx.to_pandas_adjacency(x), adj_list))

    return adj_list


def plot_net_properties(nline, ncol, properties_dict):
    """helper function providing plots for "net_properties" and "comp_network_properties (helper function)"""

    # The list of properties to plot is retrieved.
    prop_list = list(properties_dict.keys())
    max_plot = len(prop_list)
    fig, axs = plt.subplots(nline, ncol)
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.titlepad'] = -50
    l = 0
    # Plot.
    for i in range(nline):
        for j in range(ncol):
            if l < max_plot:
                # NM histogram.
                axs[i, j].hist(properties_dict[prop_list[l]][1:])
                axs[i, j].set_title(prop_list[l])  # Title
                # Red line for basic network values.
                axs[i, j].axvline(
                    properties_dict[prop_list[l]][0], color='red')
                l += 1
            else:
                axs[i, j].axis('off')
    plt.show()
    # Mean evolution.
    mean = dict.fromkeys(properties_dict)
    # The NM mean evolution is computed.
    mean = {key: [sum(properties_dict[key][1:][0:j])/len(properties_dict[key][1:][0:j])
                  for j in range(1, len(properties_dict[key][1:])+1)] for key in mean.keys()}
    # Representation.
    l = 0
    fig, axs = plt.subplots(nline, ncol)
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.titlepad'] = -50
    for i in range(nline):
        for j in range(ncol):
            if l < max_plot:
                axs[i, j].plot(mean[prop_list[l]])
                axs[i, j].set_title(prop_list[l])
                axs[i, j].axvline(200, color='red')  # trait rouge à 200
                l += 1
            else:
                axs[i, j].axis('off')
    plt.show()


def glob_properties(data, nm_list, plot=True):
    """Computes properties at the level of the global network (clustering, local efficiency).

    This function measures the global metrics for the network and each associated NM graph network. 
    The percentage of random networks with a value lower than the real network 
    for each metric is also returned as a p value.
    A graphical representation is obtained with the plot_net_properties function.

    Parameters
    ----------
    data : dataframe
        Adjacency matrix of the network of interest.
    nm_list : list of dataframes
        List of null model networks adjacency matrices.
    plot : bool, optional
        specifies whether graphic representations are expected. The default is True.

    Returns
    -------
    properties_dict : dict of list
        Contains the lists of measurements for each metric.
    p_val : dict
        Contains the percentage of random networks with a lower value than the 
        actual network for each metric (p value).

    """
    # An input list with data and then the set of random networks is created.
    input_list = [data] + nm_list
    # The adjacency matrices are converted to networkx objects.
    net_list = list(map(lambda x: nx.Graph(x), input_list))
    # Lists of metrics names and their associated functions are created.
    prop_list = ['clustering', 'local efficiency']
    func_list = [nx.average_clustering, nx.local_efficiency]
    # A list of list of values for each metrics is calculated, and the results are then stored in a dictionnary.
    value_list = list(map(lambda x: [x(i) for i in net_list], func_list))
    properties_dict = {prop: value for prop,
                       value in zip(prop_list, value_list)}
    # The dictionary containing the percentage of random networks with a value lower than the actual network for each metric is also returned.
    p_val = {key: 100 * len(list(filter(lambda x: x < properties_dict[key][0], properties_dict[key][1:]))) / len(
        properties_dict[key][1:]) for key in prop_list}

    if plot == True:
        plot_net_properties(2, 2, properties_dict)

    return properties_dict, p_val


def comp(nets, null_models, deltacon=False, plot=True):
    """Compares 2 networks (net 1 and net 2) according to global properties and 
    deltacon (if known node correspondance).

    Subtracts the values of net 2 from net 1, for both the network and the null models.
    Compares the networks (basic and null models) with deltacon if known node correspondance.
    A graphical representation is obtained with the plot_net_properties function.

    Parameters
    ----------
    nets : list of dataframe
        List containing the adjacency matrices of the networks we want to analyze.
    null_models : list of list of dataframe.
        list of list of null model networks adjacency matrices (in the same order as nets).
    deltacon : bool, optional
        Specifies whether deltacon is expected. The default is False.
    plot : bool, optional
        Specifies whether graphic representations are expected. The default is True.

    Returns
    -------
    comp_dict : dict of list
        Contains the lists of comparison measures for each metric.
    p_val : dict
        Contains the percentage of random network pairs that have a lower value than the 
        actual network for each metric.
    """
    # List of properties dict for both networks.
    prop_dicts = list(map(lambda x, y: glob_properties(
        x, y, plot=False)[0], nets, null_models))
    # Subtraction of values from net1 by net2.
    comp_dict = {key: [prop_net1 - prop_net2 for prop_net1, prop_net2 in zip(
        prop_dicts[0][key], prop_dicts[1][key])] for key in prop_dicts[0].keys()}
    # A list of input_list as in glob_properties is created.
    input_lists = [list(map(lambda x: nx.Graph(
        x), ([nets[i]] + null_models[i]))) for i in range(len(nets))]
    if deltacon == True:
        # The deltacon distance is calculated.
        comp_dict['deltacon'] = [netrd.distance.DeltaCon().dist(
            input_lists[0][i], input_lists[1][i]) for i in range(len(input_lists[0]))]
        # On convertit en score de similarité.
        comp_dict['deltacon'] = list(
            map(lambda x: 1/(x+1), comp_dict['deltacon']))
    p_val = {key: 100 * len(list(filter(lambda x: x < comp_dict[key][0], comp_dict[key][1:]))) / len(
        comp_dict[key][1:]) for key in comp_dict.keys()}

    if plot == True:
        plot_net_properties(2, 2, comp_dict)

    return comp_dict, p_val


def reciprocity(data, nm_list):
    """Compute reciprocity for directed weighted graph.

    The values are computed for the networks and associated NMs.
    A graphical representation of the representation for NM graphs and basic network.

    Parameters
    ----------
    data : dataframe
        Adjacency matrix of the directed weighted network of interest.
    nm_list : list of dataframes
        List of NM networks adjacency matrices.

    Returns
    -------
    r_list
        Lists of reciprocity measurement for the basic graph and NM.
    p_val : dict
        Contains the percentage of random networks with a lower value than the 
        actual network for each metric (p value).


    """

    def r_computation(data):
        """Computes reciprocity for one network"""

        transposed_adj = data.transpose()  # Transposed matrix.
        edge_list = data.where(
            np.triu(np.ones(data.shape)).astype(bool)).stack().reset_index()
        transposed_edge_list = transposed_adj.where(np.triu(np.ones(transposed_adj.shape)).astype(
            bool)).stack().reset_index()  # ajouter ignorer diagonale ici ?
        # The edge list is obtained with i to j and j to i values.
        full_edge_list = pd.concat((edge_list, transposed_edge_list), axis=1).set_axis(
            ['i', 'j', 'Wij', 'isuppr', 'jsuppr', 'Wji'], axis=1).drop(columns=['isuppr', 'jsuppr'])  # on colle i et j
        full_edge_list = full_edge_list.loc[full_edge_list['i']
                                            != full_edge_list['j']]
        # Edge statistics computation.
        full_edge_list.loc[:, 'Wij_bidir'] = full_edge_list[[
            'Wij', 'Wji']].min(axis=1)
        full_edge_list.loc[:, 'Wij_dir'] = full_edge_list.loc[:,
                                                              'Wij'] - full_edge_list.loc[:, 'Wij_bidir']
        full_edge_list.loc[:, 'Wji_dir'] = full_edge_list.loc[:,
                                                              'Wji']-full_edge_list.loc[:, 'Wij_bidir']
        # Node statistic computation from edge statistics.
        node_list_i = full_edge_list.groupby('i')[['Wij', 'Wji', 'Wij_bidir']].aggregate(
            np.sum).set_axis(['Sout', 'Sin', 'Sbidir'], axis=1)  # quand individu est i
        node_list_i.index.rename('indiv', True)
        node_list_j = full_edge_list.groupby('j')[['Wij', 'Wji', 'Wij_bidir']].aggregate(
            np.sum).set_axis(['Sout', 'Sin', 'Sbidir'], axis=1)  # quand individu est i
        node_list_j.index.rename('indiv', True)
        full_node_list = node_list_i.add(node_list_j, fill_value=0)
        full_node_list.loc[:, 'Sout_nonrecip'] = full_node_list.loc[:,
                                                                    'Sout'] - full_node_list.loc[:, 'Sbidir']
        full_node_list.loc[:, 'Sin_nonrecip'] = full_node_list.loc[:,
                                                                   'Sin'] - full_node_list.loc[:, 'Sbidir']
        # Full network statistic computation from node statistics.
        W = full_node_list['Sout'].sum()
        Wbidir = full_node_list['Sbidir'].sum()
        # The reciprocity is calculated from full network statistics.
        r = Wbidir/W

        return r

    # An input list with data and then the set of random networks is created.
    input_list = [data] + nm_list
    r_list = list(map(lambda x: r_computation(x), input_list))
    p_val = 100 * \
        len(list(filter(lambda x: x <
            r_list[0], r_list[1:]))) / len(r_list[1:])
    # Graphic representation.
    plt.hist(r_list[1:])  # NM histogram.
    plt.title('reciprocity')  # Title.
    plt.axvline(r_list[0], color='red')  # Basic network value representation.
    plt.show()

    return r_list, p_val
