"""This script performs the illustrative analyses on the toy data
presented in the report. I have chosen to present it in a "raw" way 
with a recommended line by line execution to allow users , especially biologists 
who are not comfortable with programming, to understand precisely each step.
The input to enter in the command line interface to reproduce the report 
figures are presented in the docstrings. Feel free to test other options!
it is recommended to look at the displayed plots in real time to get a
better understanding of the command line interface.

See the report for a detailed description.
"""

import pandas as pd
import Modules.net as net
import Modules.neuro as neuro
import Modules.etho as etho
__author__ = " Arnaud Maupas "
__contact__ = "arnaud.maupas@ens.fr"
__date__ = "19/09/22"
__version__ = "1"


"""             Ethological Data preprocessing            """


"""The cleaning is performed on Rhesus and Tonkean data with all the optional 
steps. To reproduce the report analysis on the tonkean data, please enter:
    
For the Tonkean behavioral category correction:
0:2 1:0 2:1 3:3
0:5 1:6 2:0 3:1 4:2 5:3 6:4
0:0 1:1

for the other part of the cleaning (on rhesus and tonkean):
y
400
y
3
20
"""
# Tonkean (shown in the report, Figure 7 and 8ABCD)
tonkean = etho.Focals('../Data/Etho/Raw/Tonkean',
                      open_preprocessed=False, check_empty_col=True, ignore=())
tonkean.filtering(grooming_see=True, non_visible_see=True,
                  short_focal_preprocessing_see=True, save=None, ignore=())
# Rhesus
rhesus = etho.Focals('../Data/Etho/Raw/Rhesus',
                     open_preprocessed=False, check_empty_col=True, ignore=())
rhesus.filtering(grooming_see=True, non_visible_see=True,
                 short_focal_preprocessing_see=True, save=None, ignore=())  # filtering

"""Data Visualisation"""
# Tonkean (shown in the report, Figure8EF)
tonkean.visualisation(4, 6, see_table=True)
# Rhesus
rhesus.visualisation(4, 5, see_table=True)

"""Etho network construction"""
# Rhesus affiliative networks.
rhes_adjlist, rhes_undir_adjlist, rhes_dsi, _ = rhesus.affiliative_networks_adj()
# Grooming for the reciprocity analysis.
groom = rhes_adjlist[0]
# Undirected grooming and "social play" for the network comparison.
undir_groom = rhes_undir_adjlist[0]
social_play = rhes_undir_adjlist[4]
# Tonkean dsi network.
_, _, tonk_dsi, _ = tonkean.affiliative_networks_adj()

"""pre-net NM (may take some time, you can reduce the output size with the "nb_iter" parameter (default = 1000) """
# rhesus affiliative behaviors (Figure 10A)
rhes_pre_net_adjlist, rhes_pre_net_undir_adjlist, rhes_pre_net_dsi, _ = rhesus.prenet_NM(
    seed=99)
# grooming pour analyse reciprocité (Figure 11)
nm_groom = rhes_pre_net_adjlist[0]
# undirected grooming an social play (Figure 10D)
undir_groom_nm = rhes_pre_net_undir_adjlist[0]  # Figure 11
social_play_nm = rhes_pre_net_undir_adjlist[4]
# Tonkean dsi (Figure 10A)
_, _, tonk_pre_net_dsi, _ = tonkean.prenet_NM(seed=99)


"""             Neurological Data preprocessing (Figure 9)            """

# Charge the input data.
raw_action = pd.read_csv('../Data/Neuro/action.csv', index_col=0)
raw_inter = pd.read_csv('../Data/Neuro/inter.csv', index_col=0)
# Get the filtering results.
dict_action = neuro.n_preprocessing(raw_action)
dict_inter = neuro.n_preprocessing(raw_inter)
# Charge the emprical threshold filtered data.
action = dict_action['empirical'][1]
inter = dict_inter['empirical'][1]

"""               Network analysis                         """

# An example of individual (eigenvector centrality and brokerage) and dyadic (distance) properties calculation on rhesus dsi.
centr_brok_rhes, dist_rhes = net.indiv_properties(rhes_dsi)

# The postnet NM graph are charged.
rhes_post_net_dsi = net.postnet_NM(rhes_dsi, seed=99)
tonk_post_net_dsi = net.postnet_NM(tonk_dsi, seed=99)
action_nm = net.postnet_NM(action, seed=99)
inter_nm = net.postnet_NM(inter, seed=99)


#global properties
# Tonkean (Figure 10B and C)
properties_dict_tonk_pre_net, tonk_pre_net_p_value = net.glob_properties(
    tonk_dsi, tonk_pre_net_dsi)
properties_dict_tonk_post_net, tonk_post_net_p_value = net.glob_properties(
    tonk_dsi, tonk_post_net_dsi)
# Undirected grooming and social play comparison (Figure 10 D)
properties_dict_behav, p_val_behav = net.comp(
    [undir_groom, social_play], [undir_groom_nm, social_play_nm], deltacon=True)
# DSI and neuro comparison (Figure 10 E)
properties_dict_dsi, p_val_dsi = net.comp(
    [rhes_dsi, tonk_dsi], [rhes_post_net_dsi, tonk_post_net_dsi])
properties_dict_neuro, p_val_neuro = net.comp(
    [action, inter], [action_nm, inter_nm])

# Reciprocity on grooming (Figure 11)
r_list, p_val = net.reciprocity(groom, nm_groom)
