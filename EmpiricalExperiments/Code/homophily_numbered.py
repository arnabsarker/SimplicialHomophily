#!/usr/bin/env python
# coding: utf-8

# # Homophily Calculation
# This script can be used for homophily calculations for groups with numbered group identifiers. The only parameter you should need to change for different datasets is `dataset_name` but you can change the filepath definition to load the data and change size-k if desired. 
# 
# **If you use string group identifiers:**
# - Comment out group_processing and homophily_calc and uncomment group_processing_s and homophily_calc_s
# - Change the code after `## Run Calculation` to include the \_s and the end of the functions
# 

import scipy
from scipy.stats import binom_test


import pandas as pd
from scipy.special import comb
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
import traceback
import gc
import os

## large social network tag - also applies to synthetic data
lsn = False

if(lsn):
    dataset = "soc-orkut"
    all_datasets = np.sort([i for i in os.listdir(f'../Data/{dataset}/') if i.startswith(dataset)])
else:
    all_datasets = ["cont-hospital", "cont-workplace-13", "cont-workplace-15", 
                "cont-village", "hosp-DAWN", "email-Enron", 
                "bills-senate", "bills-house", "coauth-dblp", 
                "cont-primary-school", "cont-high-school",
                "retail-trivago"]


my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

my_datasets = all_datasets[my_task_id-1:len(all_datasets):num_tasks]

## Define Functions

def create_triangle_list(G):
    '''
    Generate a data frame with a row for all closed triangles in the graph G
    '''
    elist = list(G.edges())
    num_edges = len(elist)
    num_nodes = nx.number_of_nodes(G)

    triangles = []

    for e in elist:
        # consider the elist to be in form i, j
        i, j = e

        # neigbors of i are all nodes k that appears in the list
        first_node_neighbors = set(G.neighbors(i))

        # same for node j
        second_node_neighbors = set(G.neighbors(j))

        # find intersection between those neighbors => triangle
        common_neighbors = list(first_node_neighbors & second_node_neighbors)

        for t in common_neighbors:
            curr_triangle = np.sort([i,j,t])
            triangles.append(curr_triangle)
    possible_ts = np.unique(triangles, axis=0)

    return possible_ts

def group_processing(labels, hyperedges, k):
    '''
    Inputs:
    - labels -> DataFrame with columns ['id', 'group_code'] that maps node id to group for each node
    - hyperedges -> DataFrame with each row representing a hyperedge, each column containing the node id of node_i in the hyperedge
    - k -> the size of hyperedge we are interested in (ex - k=3 is a triangle)

    Outputs:
    - groups - storage DataFrame with baseline and identification information critical for homophily calculation
    - group_typed - DataFrame with a row for each hyperedge column for each group, (row, column) value represents the type (t) of hyperedge i for group j
    '''
    # Initialize table
    groups = labels.groupby(by='group_code').count().reset_index()
    groups = groups.rename(columns = {'id':'group_size'})

    # Calculate the probability of any random draw being in each group
    N = sum(groups['group_size'])

    # Calculate the baseline scores for each type-t hyperedge (as defined in Methods A.4)
    # Here k = 3, t = 1, 2, 3
    for ind, row in groups.iterrows():
        len_X = row['group_size']
        for t_1 in range(k):
            t = t_1+1
            # Calculate and store baselines
            groups.loc[ind,'b_type-'+str(t)] = comb(len_X-1, t-1)*comb(N - len_X, k-t)/comb(N - 1, k-1)

    groups['group_code'] = groups['group_code'].astype(int)

    # Iterate through nodes in the model to relable with group id     
    label_dict = {v['id']: v['group_code'] for (k, v) in labels.to_dict(orient='index').items()}

    group_labeled = pd.DataFrame(index=hyperedges.index)
    for col in hyperedges.columns:
        group_labeled[col] = hyperedges[col].map(label_dict)

    group_typed = pd.DataFrame(index=hyperedges.index)
    unique_groups = np.sort(pd.unique(labels['group_code']))
    for i in unique_groups:
        group_typed[str(i)] = group_labeled.apply(lambda x: np.sum(x == i), axis=1)


    return groups, group_typed

def homophily_calc(groups, group_typed, closed_k, k):
    '''
    Inputs:
    - groups -> DataFrame with cols ['group_code', 'group_size', 'b_type-*'] for * = 1,...,k
    - group_typed -> DataFrame with rows representing hyperedges, each node labeled with its group number
    - closed_k -> DataFrame with rows representing len-k cycles in the graph
    - k -> the size of hyperedge we are interested in (ex - k=3 is a triangle)

    Outputs:
    - groups - output DataFrame with baseline and type 1-k homophily for each group 
    '''
    groups = groups.copy()
    # Iterate through groups and determine how many of each interaction types were observed
    for ind, row in groups.iterrows():
        group = int(row['group_code'])
        # for each t value, determine how many hyperedges are type t for the group
        for t_1 in range(k):
            t = t_1+1
            label = 'type-'+str(t)
            # compute and store frequency
            groups.loc[ind, label] = len(group_typed[group_typed[str(group)]==t])

    # set type-t to be the number of nodes in a type-t hyperedge
    for t_1 in range(k):
        t = t_1+1
        # type-t number of nodes for both groups
        groups['type-'+ str(t)] = groups['type-'+str(t)]*t


    # Determine total number of hyperedges for each group
    t_list = ['type-'+str(t+1) for t in range(k)]
    num_group = groups[t_list].sum(axis=1)

    ## Hypergraph
    # Calculate type-t affinity for all t, for both groups 
    for t_1 in range(k):
        t = t_1+1
        # type-t affinity for both groups
        groups['h_type-'+ str(t)] = (groups['type-'+str(t)])/num_group

        # affinity/baseline for both groups
        groups['h/b_type-'+str(t)] = groups['h_type-'+str(t)]/groups['b_type-'+str(t)]


    ## Simplicial Complex
    # Iterate through groups and determine how many of each interaction types were observed
    for ind, row in groups.iterrows():
        group = int(row['group_code'])
        # for each t value, determine how many hyperedges are type t for the group
        for t_1 in range(k):
            t = t_1+1
            label = 'c_type-'+str(t)
            # compute and store frequency
            groups.loc[ind, label] = len(closed_k[closed_k[str(group)]==t])

    # set type-t to be the number of nodes in a type-t hyperedge
    for t_1 in range(k):
        t = t_1+1
        # type-t number of nodes for both groups
        groups['c_type-'+ str(t)] = groups['c_type-'+str(t)]*t


    # Determine total number of hyperedges for each group
    c_t_list = ['c_type-'+str(t+1) for t in range(k)]
    num_group = groups[c_t_list].sum(axis=1)

    # Calculate type-t affinity for all t, for both groups 
    for t_1 in range(k):
        t = t_1+1
        # type-t affinity for both groups
        groups['c_h_type-'+ str(t)] = (groups['c_type-'+str(t)])/num_group

        # affinity/baseline for both groups
        groups['sc_h/b_type-'+str(t)] = groups['h_type-'+ str(t)]/groups['c_h_type-'+ str(t)]


    return groups


## Initialize (CHANGE THIS)
for dataset_name in my_datasets:
    try:

        ## Load Data (CHANGE THIS IF NEEDED)
        if(lsn):
            edges = pd.read_csv(f'../Data/{dataset}/{dataset_name}/edges.csv')
            labels = pd.read_csv(f'../Data/{dataset}/{dataset_name}/labels.csv')
            triangles = pd.read_csv(f'../Data/{dataset}/{dataset_name}/triangles.csv')
        else:
            edges = pd.read_csv(f'../Data/{dataset_name}/edges.csv')
            labels = pd.read_csv(f'../Data/{dataset_name}/labels.csv')
            triangles = pd.read_csv(f'../Data/{dataset_name}/triangles.csv')
        
        
        good_nodes = list(labels[labels['group_code'] != -1]['id'])
        labels = labels[labels['id'].isin(good_nodes)]
        edges = edges[(edges['node_1'].isin(good_nodes)) & (edges['node_2'].isin(good_nodes))]
        triangles = triangles[(triangles['node_1'].isin(good_nodes))\
                              & (triangles['node_2'].isin(good_nodes))\
                              & (triangles['node_3'].isin(good_nodes))]
        
        ## relabel groups
        labelmap = {k: i for i, k in enumerate(np.sort(pd.unique(labels['group_code'])))}
        labels['group_code'] = labels['group_code'].map(labelmap)

        ## Initialize (CHANGE THIS IF NEEDED)
        k = 3

        ## Format Data
        hyperedges = triangles[['node_1', 'node_2', 'node_3']] # filled triangles

        labels.dropna(inplace=True)


        ## Identify and Store Closed Triangles 
        if(lsn):
            tri_df = pd.read_csv(f'../Data/{dataset}/{dataset_name}/all_closed_triangles.csv')
        else:
            # Make a graph with all edges
            G = nx.from_pandas_edgelist(edges, 'node_1', 'node_2')
            G_u = G.to_undirected()

            # Identify triangles
            tri_list = create_triangle_list(G_u)
            tri_df = pd.DataFrame(tri_list, columns = ['node_1', 'node_2', 'node_3'])

        # Iterate through nodes in the model to relable with group id     
        label_dict = {v['id']: v['group_code'] for (k, v) in labels.to_dict(orient='index').items()}
        tri_group_labeled = pd.DataFrame(index=tri_df.index)
        for col in tri_df.columns:
            tri_group_labeled[col] = tri_df[col].map(label_dict)

        # Generate identifier of type (t) for each group for each hyperedge
        closed_k = pd.DataFrame(index=tri_df.index)
        for i in labels['group_code'].unique():
            closed_k[str(i)] = tri_group_labeled.apply(lambda x: np.sum(x == i), axis=1)


        ## Run Calculation
        groups, group_typed = group_processing(labels, hyperedges, k)
        groups.reset_index(drop=True, inplace=True)
        groups = homophily_calc(groups, group_typed, closed_k, k)


        ## Save output to .csv
        if(lsn):
            groups.to_csv(f'../Results/{dataset}/groups_{dataset_name}.csv')
        else:
            groups.to_csv('../Results/groups_'+dataset_name+'.csv')
        gc.collect()
    except Exception as e:
        print(dataset_name)
        print(e)
        print(traceback.format_exc())

# ## Function Definitions for string group identifiers

# def group_processing_s(labels, hyperedges, k):
#     '''
#     Inputs:
#     - labels -> DataFrame with columns ['id', 'group_code'] that maps node id to group for each node
#     - hyperedges -> DataFrame with each row representing a hyperedge, each column containing the node id of node_i in the hyperedge
#     - k -> the size of hyperedge we are interested in (ex - k=3 is a triangle)

#     Outputs:
#     - groups - storage DataFrame with baseline and identification information critical for homophily calculation
#     - group_typed - DataFrame with a row for each hyperedge column for each group, (row, column) value represents the type (t) of hyperedge i for group j
#     '''
#     # Initialize table
#     groups = labels.groupby(by='group_code').count().reset_index()
#     groups = groups.rename(columns = {'id':'group_size'})

#     # Calculate the probability of any random draw being in each group
#     N = sum(groups['group_size'])

#     # Calculate the baseline scores for each type-t hyperedge (as defined in Methods A.4)
#     # Here k = 3, t = 1, 2, 3
#     for ind, row in groups.iterrows():
#         len_X = row['group_size']
#         for t_1 in range(k):
#             t = t_1+1
#             # Calculate and store baselines
#             groups.loc[ind,'b_type-'+str(t)] = comb(len_X-1, t-1)*comb(N - len_X, k-t)/comb(N - 1, k-1)

#     # Iterate through nodes in the model to relable with group id     
#     label_dict = {v['id']: v['group_code'] for (k, v) in labels.to_dict(orient='index').items()}

#     group_labeled = pd.DataFrame(index=hyperedges.index)
#     for col in hyperedges.columns:
#         group_labeled[col] = hyperedges[col].map(label_dict)

#     group_typed = pd.DataFrame(index=hyperedges.index)
#     for i in list(groups['group_code']):
#         group_typed[str(i)] = group_labeled.apply(lambda x: np.sum(x == i), axis=1)


#     return groups, group_typed

# def homophily_calc_s(groups, group_typed, closed_k, k):
#     '''
#     Inputs:
#     - groups -> DataFrame with cols ['group_code', 'group_size', 'b_type-*'] for * = 1,...,k
#     - group_typed -> DataFrame with rows representing hyperedges, each node labeled with its group number
#     - closed_k -> DataFrame with rows representing len-k cycles in the graph
#     - k -> the size of hyperedge we are interested in (ex - k=3 is a triangle)

#     Outputs:
#     - groups - output DataFrame with baseline and type 1-k homophily for each group 
#     '''
#     # Iterate through groups and determine how many of each interaction types were observed
#     for ind, row in groups.iterrows():
#         group = row['group_code']
#         # for each t value, determine how many hyperedges are type t for the group
#         for t_1 in range(k):
#             t = t_1+1
#             label = 'type-'+str(t)
#             # compute and store frequency
#             groups.loc[ind, label] = len(group_typed[group_typed[group]==t])

#     # set type-t to be the number of nodes in a type-t hyperedge
#     for t_1 in range(k):
#         t = t_1+1
#         # type-t number of nodes for both groups
#         groups['type-'+ str(t)] = groups['type-'+str(t)]*t


#     # Determine total number of hyperedges for each group
#     t_list = ['type-'+str(t+1) for t in range(k)]
#     num_group = groups[t_list].sum(axis=1)

#     ## Hypergraph
#     # Calculate type-t affinity for all t, for both groups 
#     for t_1 in range(k):
#         t = t_1+1
#         # type-t affinity for both groups
#         groups['h_type-'+ str(t)] = (groups['type-'+str(t)])/num_group

#         # affinity/baseline for both groups
#         groups['h/b_type-'+str(t)] = groups['h_type-'+str(t)]/groups['b_type-'+str(t)]


#     ## Simplicial Complex
#     # Iterate through groups and determine how many of each interaction types were observed
#     for ind, row in groups.iterrows():
#         group = row['group_code']
#         # for each t value, determine how many hyperedges are type t for the group
#         for t_1 in range(k):
#             t = t_1+1
#             label = 'c_type-'+str(t)
#             # compute and store frequency
#             groups.loc[ind, label] = len(closed_k[closed_k[group]==t])

#     # set type-t to be the number of nodes in a type-t hyperedge
#     for t_1 in range(k):
#         t = t_1+1
#         # type-t number of nodes for both groups
#         groups['c_type-'+ str(t)] = groups['c_type-'+str(t)]*t


#     # Determine total number of hyperedges for each group
#     c_t_list = ['c_type-'+str(t+1) for t in range(k)]
#     num_group = groups[c_t_list].sum(axis=1)

#     # Calculate type-t affinity for all t, for both groups 
#     for t_1 in range(k):
#         t = t_1+1
#         # type-t affinity for both groups
#         groups['c_h_type-'+ str(t)] = (groups['c_type-'+str(t)])/num_group

#         # affinity/baseline for both groups
#         groups['sc_h/b_type-'+str(t)] = groups['h_type-'+ str(t)]/groups['c_h_type-'+ str(t)]


#     return groups