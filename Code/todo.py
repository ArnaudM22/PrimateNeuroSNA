"""This script contains the raw material for further development 
presented in the report (part IV.3). It is still a draft, thus not clean at all
(computation performed in the global space, comments in French, very long computation time...)

For subsampling, it is very long to compute. n on the x absis is the randomly drawn number of focal
per individual. The populations are of size 1000. 

For analytical Null Model, see the wrg table and ref [47] (especially the SI). 
The wrg have been used to compute rho (and its standard deviation) analytically 
for each affiliative directed behaviors. It uses a Jacknife resampling method.


See the report for a detailed description.
"""

import Modules.net as net
import Modules.etho as etho
import numpy as np
import networkx as nx
import netrd
import random
import pandas as pd
import matplotlib.pyplot as plt
__author__ = " Arnaud Maupas "
__contact__ = "arnaud.maupas@ens.fr"
__date__ = "19/09/22"
__version__ = "1"


# Opening data and creating variable to run the analyses on the global space.
rhesus = etho.Focals('../Data/Etho/preprocessed_rhesus.csv')
names = rhesus.indiv
affiliative_behaviors = rhesus.affiliative_behaviors
focals = rhesus.data
adjlist, _, dsi, _ = rhesus.affiliative_networks_adj()
dsinet = nx.Graph(dsi)

# Charging both null model (warning, a little bit long because of pre-net NM)
_, _, pre_net, _ = rhesus.prenet_NM()
post_net = net.postnet_NM(dsi)

"""                  Subsampling                       """

# DSI computation (called later)


def dsi_func(focals):
    """Computes the dsi from a focal dataset (helper function)"""
    tempsobsind1 = (focals.groupby(['numfocal', 'Subject']).last(
    ).loc[:, 'focal_length']).groupby('Subject').sum()  # Individual observation length
    tempsobsind = pd.Series(0, index=names, name='focal_length').combine(
        tempsobsind1, lambda s1, s2: s1 if s1 > s2 else s2, fill_value=0)
    # Dyad observation length.
    tempsobsdyad = pd.DataFrame(0, index=names, columns=names)
    for i in range(len(tempsobsdyad)):
        tempsobsdyad.iloc[i] = tempsobsind + tempsobsind.iloc[i]
    tempsobstot = tempsobsdyad.to_numpy().sum()/2  # Total observation length
    tempsobscomp1 = focals.loc[focals['Behavior'].isin(affiliative_behaviors)].groupby(
        'Behavior')['Duration (s)'].sum()  # Behaviors observation length.
    tempsobscomp = pd.Series(0, index=affiliative_behaviors, name='focal_length').combine(
        tempsobscomp1, lambda s1, s2: s1 if s1 > s2 else s2, fill_value=0)
    # Undirected case.
    adjnonoriente = pd.DataFrame(focals.loc[focals['Behavior'].isin(['Etreinte', 'Jeu social', 'Contact passif'])].groupby(
        ['Behavior', 'Subject', 'Modifiers'])['Duration (s)'].sum())  # duree pour chaque comportement, chaque individu, et chaque modifiers different
    if adjnonoriente.empty == False:
        adjnonoriente = adjnonoriente.merge(pd.Series(adjnonoriente.index.get_level_values('Modifiers'), index=adjnonoriente.index).str.split(
            ',', expand=True), left_index=True, right_index=True)  # separer les individus dans modifiers
        adjnonoriente2 = adjnonoriente.loc[:, ['Duration (s)', 0]].dropna().rename(
            columns={0: 'Modifiers'})  # creer table qui va servir dans la boucle
        # concatener les tables de duree pour les individus dans modifiers
        for i in range(1, len(adjnonoriente.columns)-1):
            adjnonoriente2 = pd.concat((adjnonoriente2, adjnonoriente.loc[:, [
                                       'Duration (s)', i]].dropna().rename(columns={i: 'Modifiers'})))
        # enlever ancien modifiers et calculer les valeurs sommées pour chaque individu dans Modifiers
        adjnonoriente = adjnonoriente2.set_index(adjnonoriente2.index.droplevel(
            2)).groupby(['Behavior', 'Subject', 'Modifiers']).sum()
    # cas oriente
    adjoriente = pd.DataFrame(focals.loc[focals['Behavior'].isin(['Portage', 'Se repose sur', 'Grooming'])].groupby(
        ['Behavior', 'Subject', 'Modifiers'])['Duration (s)'].sum())  # duree pour chaque comportement, chaque individu, et chaque modifiers different
    if adjoriente.empty == False:
        adjoriente = adjoriente.merge(pd.Series(adjoriente.index.get_level_values('Modifiers'), index=adjoriente.index).str.split('|', expand=True), left_index=True, right_index=True).rename(
            columns={0: 'direction', 1: 'Modifiers'}).set_index(adjoriente.index.droplevel(2))  # separer les donnees dans modifiers
        adjoriente.loc[:, 'Subject'] = adjoriente.index.get_level_values(
            1)  # on cree colonne Subject
        adjoriente = adjoriente.replace('None', np.nan).dropna(
            axis=0)  # on supprime lignes avec None
        adjoriente.loc[adjoriente['direction'] == 'Focal est recepteur', ['Modifiers', 'Subject']] = adjoriente.loc[adjoriente['direction']
                                                                                                                    == 'Focal est recepteur', ['Subject', 'Modifiers']].values  # on inverse Subject et Modifiers pour avoir tjrs l'emtteur en suject
        # on supprime la colonne d'index Subject remplacée par colonne Subject
        adjoriente = adjoriente.set_index(adjoriente.index.droplevel(1))
        # calculer les valeurs sommées pour chaque individu dans Modifiers
        adjoriente = adjoriente.groupby(
            ['Behavior', 'Subject', 'Modifiers']).sum()
    # calcul matrice adjacence totale
    iterables = [affiliative_behaviors, names, names]  # contenu index
    index = pd.MultiIndex.from_product(
        iterables, names=['Behavior', 'Subject', 'Modifiers'])  # initaliser index
    adj = pd.DataFrame(index=index).merge(pd.concat((adjnonoriente, adjoriente)), how='left', right_index=True, left_index=True).fillna(
        0).unstack(1)  # remplir dataframe avec la valeur des tables orientes et nonoriente concatenees
    # supprimer column multi index qui sert à rien
    adj.columns = adj.columns.droplevel()
    for i in ['Etreinte', 'Jeu social', 'Contact passif']:
        # symetriser comportement non oriente
        adj.loc[i] = (adj.loc[i].transpose()+adj.loc[i]).values
    # diviser par duree d'observation dyade
    for i in affiliative_behaviors:
        adj.loc[i] = adj.loc[i].div(tempsobsdyad).values
    # partie DSI
    predsi = adj.copy(deep=True)
    for i in ['Portage', 'Se repose sur', 'Grooming']:
        # symetriser comportement oriente
        predsi.loc[i] = (predsi.loc[i].transpose()+predsi.loc[i]).values
    predsi = predsi.fillna(0)
    dsi = pd.DataFrame(0, index=names, columns=names)  # initialisation
    for i in affiliative_behaviors:
        if tempsobscomp[i] != 0:
            dsi += (predsi.loc[i]/(tempsobscomp/tempsobstot)
                    [i])/len(affiliative_behaviors)

    dsi = nx.Graph(dsi)
    return dsi


# on prepare liste de focale pour chaque individu
nbfocalind = pd.DataFrame(focals.groupby('numfocal').first().loc[:, 'Subject'])
nbfocalind.loc[:, 'numfocal2'] = nbfocalind.index
nbfocalind = nbfocalind.groupby('Subject')['numfocal2'].apply(list)

# on prepare dict de valeurs
listfoc = {k: []
           for k in range(1, min(nbfocalind.apply(lambda row: len(row)))+1)}
focalsiter = {k: []
              for k in range(1, min(nbfocalind.apply(lambda row: len(row)))+1)}
deltaconsamp = {k: []
                for k in range(1, min(nbfocalind.apply(lambda row: len(row)))+1)}
for i in range(1, min(nbfocalind.apply(lambda row: len(row)))+1):
    for j in range(1000):
        listfoc[i].append(nbfocalind.apply(
            lambda row: random.choices(row, k=i)))
    listfoc[i] = list(map(lambda x: [item for sublist in x.tolist()
                      for item in sublist], listfoc[i]))
    focalsiter[i] = list(
        map(lambda x: focals.loc[focals['numfocal'].isin(x)], listfoc[i]))
    deltaconsamp[i] = list(map(
        lambda x: 1/(netrd.distance.DeltaCon().dist(dsinet, dsi_func(x))+1), focalsiter[i]))


plt.figure(figsize=(14, 6))
plt.boxplot(deltaconsamp.values(), positions=list(range(1, 15)))
plt.show()

post_net_graph = list(map(lambda x: nx.Graph(x), post_net))
delta_post_net = list(
    map(lambda x: 1/(netrd.distance.DeltaCon().dist(dsinet, x)+1), post_net_graph))
pre_net_graph = list(map(lambda x: nx.Graph(x), pre_net))
delta_pre_net = list(
    map(lambda x: 1/(netrd.distance.DeltaCon().dist(dsinet, x)+1), pre_net_graph))

deltaconsamp2 = deltaconsamp
# ajouter au sample
deltaconsamp2['post_net'] = delta_post_net
deltaconsamp2['pre_net'] = delta_pre_net

plt.figure(figsize=(16, 6))
plt.boxplot(deltaconsamp2.values(), positions=list(
    range(1, 17)), labels=deltaconsamp2.keys())
plt.show()


"""                  Reciprocity                       """


# on sauve la liste des comportements orientes
comporiente = [affiliative_behaviors[i] for i in [0, 2, 3]]
# analyse avec methode article
oriente = [adjlist[i] for i in [0, 2, 3]]
orientebis = list(map(lambda x: x.transpose(), oriente))
# get ij list
edge = list(map(lambda x: x.where(np.triu(np.ones(x.shape)).astype(
    bool)).stack().reset_index(), oriente))  # on cree edge list triang supp
edgebis = list(map(lambda x: x.where(np.triu(np.ones(x.shape)).astype(
    bool)).stack().reset_index(), orientebis))  # idem pour transpo
for j in range(len(comporiente)):  # boucle pour creer index par comportement
    edge[j].index = [comporiente[j]]*len(edge[j])
    edgebis[j].index = [comporiente[j]]*len(edgebis[j])
# on cree table unique avec index permettant de departager
edge = pd.concat(edge)
edgebis = pd.concat(edgebis)
# on attribue arbitrairement un i et un j à chaque indiv dans le lien
edgearticle = pd.concat((edge, edgebis), axis=1).set_axis(
    ['i', 'j', 'Wij', 'isuppr', 'jsuppr', 'Wji'], axis=1).drop(columns=['isuppr', 'jsuppr'])  # on colle i et j
edgearticle = edgearticle.loc[edgearticle['i'] != edgearticle['j']]
edgearticle.index.name = 'Comp'
# calcul des stats edge
edgearticle.loc[:, 'Wij_bidir'] = edgearticle[['Wij', 'Wji']].min(axis=1)
edgearticle.loc[:, 'Wij_dir'] = edgearticle.loc[:, 'Wij'] - \
    edgearticle.loc[:, 'Wij_bidir']
edgearticle.loc[:, 'Wji_dir'] = edgearticle.loc[:, 'Wji'] - \
    edgearticle.loc[:, 'Wij_bidir']
# analyse propriete nodes
nodearticlei = edgearticle.groupby(['Comp', 'i'])['Wij', 'Wji', 'Wij_bidir'].aggregate(
    np.sum).set_axis(['Sout', 'Sin', 'Sbidir'], axis=1)  # quand individu est i
nodearticlei.index.rename('indiv', 1, True)
nodearticlej = edgearticle.groupby(['Comp', 'j'])['Wij', 'Wji', 'Wij_bidir'].aggregate(np.sum).set_axis(
    ['Sin', 'Sout', 'Sbidir'], axis=1).reindex(columns=['Sout', 'Sin', 'Sbidir'])  # quand individu est j
nodearticlej.index.rename('indiv', 1, True)
nodearticle = nodearticlei.add(nodearticlej, fill_value=0)
nodearticle.loc[:, 'Sout_nonrecip'] = nodearticle.loc[:,
                                                      'Sout']-nodearticle.loc[:, 'Sbidir']
nodearticle.loc[:, 'Sin_nonrecip'] = nodearticle.loc[:,
                                                     'Sin']-nodearticle.loc[:, 'Sbidir']
# mesures à l'echelle du réseau
W = nodearticle.groupby('Comp')['Sout'].sum()
Wbidir = nodearticle.groupby('Comp')['Sbidir'].sum()
r = Wbidir/W
r = r.to_frame(name='r')
r.loc[:, 'std'] = np.nan
# resampling
preJ = edgearticle.loc[:, ['i', 'j', 'Wij', 'Wji']]
jacknife = {k: [] for k in preJ.index}
for y, i in preJ.iterrows():
    if i['Wij'] != 0:
        t = preJ.loc[y]
        t.loc[(t['i'] == i['i']) & (t['j'] == i['j']), 'Wij'] = 0
        jacknife[y].append(t)
    if i['Wji'] != 0:
        t = preJ.loc[y]
        t.loc[(t['i'] == i['i']) & (t['j'] == i['j']), 'Wji'] = 0
        jacknife[y].append(t)
for i in jacknife.keys():
    list(map(lambda x: x.insert(4, 'Wij_bidir',
         x[['Wij', 'Wji']].min(axis=1)), jacknife[i]))
# Calcul des stats pour chaque noeud
jacknifenode = {k: [] for k in preJ.index}
for i in jacknifenode.keys():
    jacknifenodei = list(map(lambda x: x.groupby('i')['Wij', 'Wji', 'Wij_bidir'].aggregate(
        np.sum).set_axis(['Sout', 'Sin', 'Sbidir'], axis=1), jacknife[i]))
    list(map(lambda x: x.index.rename('indiv', True), jacknifenodei))
    jacknifenodej = list(map(lambda x: x.groupby('j')['Wij', 'Wji', 'Wij_bidir'].aggregate(
        np.sum).set_axis(['Sout', 'Sin', 'Sbidir'], axis=1), jacknife[i]))
    list(map(lambda x: x.index.rename('indiv', True), jacknifenodej))
    jacknifenode[i] = [x.add(y, fill_value=0)
                       for x, y in zip(jacknifenodei, jacknifenodej)]

jacknifeR = {k: [] for k in preJ.index}
for i in jacknifeR.keys():
    jacknifeR[i] = list(map(lambda x: (x.loc[:, 'Sbidir'].sum()) /
                        (x.loc[:, 'Sout'].sum()), jacknifenode[i]))
    r.loc[i, 'std'] = ((len(jacknifeR[i])-1)**(0.5))*np.std(jacknifeR[i])

fig, axs = plt.subplots(1, 3)  # initialiser figure
for i in range(3):
    print(jacknifeR[list(jacknifeR.keys())[i]])
    axs[i].hist(jacknifeR[list(jacknifeR.keys())[i]])
    axs[i].set_title(list(jacknifeR.keys())[i])
    axs[i].axvline(r.loc[list(jacknifeR.keys())[i], 'r'], color='red')
plt.show()
# WRG
wrg = pd.DataFrame()
wrg.loc[:, 'p_etoile'] = W/(W + (len(names)*(len(names)-1)))
wrg.loc[:, 'rWRG'] = wrg.loc[:, 'p_etoile'] / (1+wrg.loc[:, 'p_etoile'])
wrg.loc[:, 'rhoWRG'] = (r.loc[:, 'r']-wrg.loc[:, 'rWRG']
                        )/(1-wrg.loc[:, 'rWRG'])
wrg.loc[:, 'varrhoWRG'] = (r.loc[:, 'std']**2)/((1-wrg.loc[:, 'rWRG'])**2)
wrg.loc[:, 'stdrhoWRG'] = wrg.loc[:, 'varrhoWRG'] ** (0.5)
