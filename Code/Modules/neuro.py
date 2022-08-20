"""This module contains tools for the preprocessing of cerebral networks constructed from fMRI data.

It contains only the "n_preprocessing" function, which allows to analyze the 
efficiency cost optimization criterion on the data before filtering them.

See the "n_preprocessing" docstring and the report for a detailed description.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
__author__ = " Arnaud Maupas "
__contact__ = "arnaud.maupas@ens.fr"
__date__ = "19/09/22"
__version__ = "1"


def n_preprocessing(data):
    """Allows to analyze the Efficiency Cost Optimization criterion on the data before filtering them.

    Reproduction of the ECO analysis on our data (see report).
    It works by starting from an empty network (our network from which we 
    removed all the links), then adding in turn the links (from the strongest to the weakest) 
    while binarizing them. At each step, we take the values of EL, EG and J. 
    The optimum is calculated and compared to the theoretical one. A graphical representation is then obtained.
    The adjacency matrices with each associated threshold are returned to the user.

    Parameters
    ----------
    data : dataframe
        Adjacency matrix of the network to be analyzed.

    Returns
    -------
    thresholded_net : dict of list
        Threshold (first element) and adjacency matrix (second element) for 
        each threshold (including no threshold, i.e the original dataset.).

    """
    # The node is retrieved.
    node_list = list(data.columns)
    # The edge list is retrieved.
    # Triangular matrix.
    edge_list = data.where(np.triu(np.ones(data.shape)).astype(bool))
    np.fill_diagonal(edge_list.values, np.nan)  # The diagonal is emptied.
    edge_list = edge_list.stack().reset_index()
    edge_list.columns = ['i', 'j', 'weight']
    edge_list = edge_list.sort_values(
        "weight", ascending=False, ignore_index=True)
    construction_edge_list = edge_list.copy(deep=True)
    # Initialization of the network modified by successive iterations.
    construction_net = nx.Graph()
    # Adding the nodes.
    construction_net.add_nodes_from(node_list)
    # Initialization of the table containing the metrics at each successive iteration.
    Jdatathreshold = pd.DataFrame(columns=['Density', 'GE', 'GL'])
    # Initialization of the density and the number of iterations.
    dens = n = 0
    while dens != 1:
        # The maximum edge value is is deleted from the construction edge list.
        construction_edge_list.loc[n, 'weight'] = np.nan
        # An edge with the same i and j is created in the construction graph.
        construction_net.add_edge(
            construction_edge_list.loc[n, 'i'], construction_edge_list.loc[n, 'j'])
        # The metrics are computed.
        Jdatathreshold.loc[n, 'Density'] = dens = nx.density(construction_net)
        Jdatathreshold.loc[n, 'GE'] = nx.global_efficiency(construction_net)
        Jdatathreshold.loc[n, 'GL'] = nx.local_efficiency(construction_net)
        # Iteration level is updated.
        n += 1
    # J computation.
    Jdatathreshold.loc[:, 'J'] = (
        Jdatathreshold.loc[:, 'GE'] + Jdatathreshold.loc[:, 'GL']) / Jdatathreshold.loc[:, 'Density']
    Jdatathreshold.loc[:, 'Jnorm'] = Jdatathreshold.loc[:,
                                                        'J'] / Jdatathreshold.loc[:, 'J'].max()
    # Threshold computation.
    thresholded_net = {'dens_basic': [nx.density(nx.Graph((data != 0).astype(int))), data],
                       'theor': [3 / (len(node_list) - 1), None],
                       'empirical': [float(Jdatathreshold.loc[Jdatathreshold.loc[:, 'Jnorm'] == 1, 'Density']), None]}
    # Filtered networks construction.
    for threshold in ['theor', 'empirical']:
        n_edge = int((len(node_list) * (len(node_list) - 1)
                     * thresholded_net[threshold][0])/2)
        thresholded_edge_list = edge_list.head(n_edge)  # Edge list.
        # Graph conversion.
        thresholded_graph = nx.from_pandas_edgelist(
            thresholded_edge_list, 'i', 'j', 'weight')
        # Thresholding.
        thresholded_net[threshold][1] = nx.to_pandas_adjacency(
            thresholded_graph)
    #  Graphical representation.
    plt.plot(Jdatathreshold.loc[:, 'Density'],
             Jdatathreshold.loc[:, 'GE'], 'b')
    plt.plot(Jdatathreshold.loc[:, 'Density'],
             Jdatathreshold.loc[:, 'GL'], 'r')
    plt.plot(Jdatathreshold.loc[:, 'Density'],
             Jdatathreshold.loc[:, 'Jnorm'], 'k')
    # Thresholds are plotted.
    plt.axvline(thresholded_net['theor'][0], color='grey', linestyle='dotted')
    plt.axvline(thresholded_net['empirical'][0],
                color='grey', linestyle='dashdot')
    plt.axvline(thresholded_net['dens_basic'][0], color='grey')
    plt.show()

    return thresholded_net
