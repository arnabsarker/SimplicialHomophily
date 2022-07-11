import pandas as pd
import networkx as nx
import numpy as np
import sys
import os
from common import create_triangle_list

## large social network tag - also applies to synthetic data
lsn = True

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

results = []
for dataset_name in my_datasets:
    
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

    
    ## Identify and Store Closed Triangles 
    if(lsn):
        closed_tdf = pd.read_csv(f'../Data/{dataset}/{dataset_name}/all_closed_triangles.csv')
    else:
        ## get dataframe of closed triangles
        G = nx.from_pandas_edgelist(edges_df, 'node_1', 'node_2')
        closed_tlist = create_triangle_list(G)
        closed_tdf = pd.DataFrame(closed_tlist)
        closed_tdf.columns = ['node_1', 'node_2', 'node_3']
    
    ## map each node to its associated label
    tlabels_df = triangles_df.applymap(lambda x: labels_dict[x] if x in labels_dict else np.nan)
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
        'observed_proportion': obs,
        'closed_baseline': simp_b,
        'node_baseline': node_b,
        'simplicial_ratio': obs / simp_b,
        'hypergraph_ratio': obs / node_b,
        'number filled (hom, total)': (num_hom_filled, tot_filled),
        'number closed (hom, total)': (num_hom_closed, tot_closed)
    })

    results_df = pd.DataFrame(results)
    ## Save output to .csv
    if(lsn):
        results_df.to_csv(f'../Results/{dataset}/global_homophily_part_{my_task_id}.csv')
    else:
        results_df.to_csv(f'../Results/global_homophily_part_{my_task_id}.csv')
    