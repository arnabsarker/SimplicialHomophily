import pandas as pd
import networkx as nx
import numpy as np
import sys
import os
from common import create_triangle_list

## large social network tag - also applies to synthetic data
lsn = True

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

results = []
for dataset_name in my_datasets:
    
    ## Load Data
    if(lsn):
        edges_df = pd.read_csv(f'../Data/{dataset}/{dataset_name}/edges.csv')
        labels_df = pd.read_csv(f'../Data/{dataset}/{dataset_name}/labels.csv')
    else:
        edges_df = pd.read_csv(f'../Data/{dataset_name}/edges.csv')
        labels_df = pd.read_csv(f'../Data/{dataset_name}/labels.csv')
    
    labels_dict = {row['id']: row['group_code'] for i, row in labels_df.iterrows()}

    ## map each node to its associated label
    elabels_df = edges_df.applymap(lambda x: labels_dict[x] if x in labels_dict else np.nan)
    elabels_df = elabels_df.dropna()

    ## get counts of homophilous edges and ratio
    num_hom_edges = np.sum((elabels_df['node_1'] == elabels_df['node_2']))
    tot_edges = len(elabels_df)
    obs = num_hom_edges / tot_edges
    
    ## baseline
    group_counts = labels_df.groupby('group_code').count()
    node_b = 0
    total_nodes = len(labels_df)
    for code, ct in group_counts.iterrows():
        num = ct['id']
        node_b += float(num) * (num - 1) / (total_nodes * (total_nodes - 1) )
    
    results.append({
        'dataset': dataset_name,
        'observed_proportion': obs,
        'node_baseline': node_b,
        'edge_homophily_ratio': obs / node_b
    })

    results_df = pd.DataFrame(results)
    ## Save output to .csv
    if(lsn):
        results_df.to_csv(f'../Results/{dataset}/edge_homophily_part_{my_task_id}.csv')
    else:
        results_df.to_csv(f'../Results/edge_homophily_part_{my_task_id}.csv')
    