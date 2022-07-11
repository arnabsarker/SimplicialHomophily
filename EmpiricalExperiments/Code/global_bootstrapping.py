import pandas as pd
from scipy.special import comb
import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from common import create_triangle_list


## large social network tag
lsn = False

if(lsn):
    dataset = "soc-livejournal"
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

## Initialize (CHANGE THIS)
for dataset_name in my_datasets:
#     try:
    ## Load Data
    if(lsn):
        edges_df = pd.read_csv(f'../Data/{dataset}/{dataset_name}/edges.csv')
        labels_df = pd.read_csv(f'../Data/{dataset}/{dataset_name}/labels.csv')
        triangles_df = pd.read_csv(f'../Data/{dataset}/{dataset_name}/triangles.csv')
    else:
        edges_df = pd.read_csv(f'../Data/{dataset_name}/edges.csv')
        labels_df = pd.read_csv(f'../Data/{dataset_name}/labels.csv')
        triangles_df = pd.read_csv(f'../Data/{dataset_name}/triangles.csv')

    labels_dict = {row['id']: row['group_code'] for i, row in labels_df.iterrows()}

    # Loop through groups
    results = []
    for ite in range(100):
        # Define subset of nodes to use
    #     node_subset =labels.sample(frac=0.75)
        nodes = list(labels_df.groupby('group_code')['id'].apply(lambda s: s.sample(frac=0.75)))
        node_subset = labels_df[labels_df['id'].isin(nodes)]

        ## take subset
        edges_subset = edges_df[(edges_df['node_1'].isin(nodes)) & (edges_df['node_2'].isin(nodes))].copy()
        triangles_subset = triangles_df[(triangles_df['node_1'].isin(nodes))\
                          & (triangles_df['node_2'].isin(nodes))\
                          & (triangles_df['node_3'].isin(nodes))].copy()

        ## Identify and Store Closed Triangles 
        G = nx.from_pandas_edgelist(edges_subset, 'node_1', 'node_2')
        closed_tlist = create_triangle_list(G)
        closed_tdf = pd.DataFrame(closed_tlist)
        closed_tdf.columns = ['node_1', 'node_2', 'node_3']

        ## map each node to its associated label
        tlabels_df = triangles_subset.applymap(lambda x: labels_dict[x] if x in labels_dict else np.nan)
        tlabels_df = tlabels_df.dropna()
        closed_tlabels_df = closed_tdf.applymap(lambda x: labels_dict[x] if x in labels_dict else np.nan)
        closed_tlabels_df = closed_tlabels_df.dropna()

        ## get counts of homophilous filled triangles
        num_hom_filled = np.sum((tlabels_df['node_1'] == tlabels_df['node_2']) & \
                     (tlabels_df['node_2'] == tlabels_df['node_3']))
        tot_filled = len(tlabels_df)
        obs = num_hom_filled / tot_filled

        ## get counts of homophilous closed triangles
        num_hom_closed = np.sum((closed_tlabels_df['node_1'] == closed_tlabels_df['node_2']) & \
                        (closed_tlabels_df['node_2'] == closed_tlabels_df['node_3'])) 
        tot_closed = len(closed_tlabels_df)
        simp_b = num_hom_closed / tot_closed


        ## naive computation of node homophily baseline
        group_counts = labels_df.groupby('group_code').count()
        node_b = 0
        total_nodes = len(labels_df)
        for code, ct in group_counts.iterrows():
            num = ct['id']
            node_b += float(num) * (num - 1) * (num - 2) / (total_nodes * (total_nodes - 1) * (total_nodes - 2))

        results.append({
            'dataset': dataset_name,
            'trial': ite,
            'observed_proportion': obs,
            'closed_baseline': simp_b,
            'node_baseline': node_b,
            'simplicial_ratio': obs / simp_b,
            'hypergraph_ratio': obs / node_b,
            'number filled (hom, total)': (num_hom_filled, tot_filled),
            'number closed (hom, total)': (num_hom_closed, tot_closed)
        })

        results_df = pd.DataFrame(results)
        if(lsn):
            results_df.to_csv(f'../Results/{dataset}/gbootstrap_{dataset_name}.csv')
        else:
            results_df.to_csv('../Results/gbootstrap_'+dataset_name+'.csv')
            
#     except Exception as e:
#         print(dataset_name)
#         print(e)