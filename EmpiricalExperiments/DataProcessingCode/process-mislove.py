import pandas as pd
import numpy as np
import networkx as nx
import os
import dask.dataframe as dd
import sys
from common import create_triangle_list

my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

dataset = 'orkut'
k = 10 # number of groups to consider
num_per_subset = 20000 # number of nodes in each subset
final_set = set()
    
if((dataset == "orkut") or (dataset == "livejournal") ):
    # found largest groups through separate script
    if (dataset == "orkut"):
        groups = [ 6788,  8686,  6899,  6341,  6333,  6372, 34648, 34014,  6786, 26630]
    if (dataset == "livejournal"):
        groups = [113, 27, 324, 49, 87, 2, 123, 12, 211, 7]
    
    membership_iter = pd.read_csv(f'../RawData/{dataset}/{dataset}-groupmemberships.txt', 
                            sep='\t', names=["node", "group"], iterator=True, chunksize=10000)
    group_df = pd.concat([chunk[chunk['group'].isin(groups)] for chunk in membership_iter])
    groupmember_df = group_df.groupby("group")['node'].apply(set).reset_index()
    groupmember_df['size'] = groupmember_df['node'].apply(len)
    sorted_df = groupmember_df.sort_values(by='size', ascending=False)
    for group in sorted_df['node'].iloc[0:10]:
        final_set = final_set.union(group)
else:
    membership_df = pd.read_csv(f'../RawData/{dataset}/{dataset}-groupmemberships.txt', 
                            sep='\t', names=["node", "group"])
    groupmember_df = membership_df.groupby("group")['node'].apply(set).reset_index()
    groupmember_df['size'] = groupmember_df['node'].apply(len)
    sorted_df = groupmember_df.sort_values(by='size', ascending=False)

    final_set = set()
    groups = sorted_df.iloc[:k]['group']
    for group in sorted_df.iloc[:k]['node']:
        final_set = final_set.union(group)

all_users = list(final_set)

np.random.shuffle(all_users)
user_splits = np.array_split(all_users, np.ceil(len(all_users) / num_per_subset))

my_idx = [i for i in range(my_task_id-1, len(user_splits), num_tasks)]
my_splits = user_splits[my_task_id-1:len(user_splits):num_tasks]

for idx, split in zip(my_idx, my_splits):
    try:
        output_dir = f'../Data/soc-{dataset}/{dataset}-{idx}'
        os.makedirs(output_dir, exist_ok=True)
        iter_csv = pd.read_csv(f"../RawData/{dataset}/{dataset}-links.txt", 
                               iterator=True, chunksize=1000, 
                               sep='\t', names=['node_1', 'node_2'])
        links_df = pd.concat([chunk[(chunk['node_1'].isin(split)) & (chunk['node_2'].isin(split))] \
                              for chunk in iter_csv])
        links_df = links_df.reset_index(drop=True)
        links_df.to_csv(f"{output_dir}/edges.csv", index=False)

        group_iter_csv = pd.read_csv(f'../RawData/{dataset}/{dataset}-groupmemberships.txt',     
                                        iterator=True, chunksize=1000, 
                                        sep='\t', names=['node', 'group'])
        groupmembership_df = pd.concat([chunk[chunk['node'].isin(split) & \
                                          chunk['group'].isin(groups)] for chunk in group_iter_csv])

        groups_dict = groupmembership_df.groupby('node').apply(lambda x: list(x['group'])).to_dict()

        G = nx.from_pandas_edgelist(links_df, 'node_1', 'node_2')
        all_tlist = create_triangle_list(G)

        final_tlist = []
        for (i,j,k) in all_tlist:
            all_groups_i = set(groups_dict.setdefault(i, []))
            all_groups_j = set(groups_dict.setdefault(j, []))
            all_groups_k = set(groups_dict.setdefault(k, []))
            common_groups = (all_groups_i & all_groups_j) & all_groups_k
            if(len(common_groups) > 0):
                final_tlist.append((i,j,k))

        allt_df = pd.DataFrame(all_tlist)
        allt_df.columns = ['node_1', 'node_2', 'node_3']
        allt_df.to_csv(f"{output_dir}/all_closed_triangles.csv", index=False)

        t_df = pd.DataFrame(final_tlist)
        t_df.columns = ['node_1', 'node_2', 'node_3']
        t_df.to_csv(f"{output_dir}/triangles.csv", index=False)

        labels = {}
        for node, neighbor_df in links_df.groupby('node_1'):
            all_neighbors = list(neighbor_df['node_2'])
            node_groups = groups_dict.setdefault(node, [])
            if len(node_groups) == 0:
                labels[node] = -1
            elif len(node_groups) == 1:
                labels[node] = node_groups[0]
            else:
                counts = {k: 0 for k in node_groups}
                for neighbor in all_neighbors:
                    neighbor_groups = groups_dict.setdefault(neighbor, [])
                    for g in node_groups:
                        if g in neighbor_groups:
                            counts[g] = counts[g] + 1
                labels[node] = max(counts, key=counts.get)
        label_df = pd.DataFrame.from_dict(labels, orient='index').reset_index()
        label_df.columns = ['id', 'group_code']
        label_df.to_csv(f"{output_dir}/labels.csv", index=False)
    except Exception as e:
        print(idx)
        print(e)
