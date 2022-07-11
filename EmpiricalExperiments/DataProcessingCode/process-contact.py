import pandas as pd
import numpy as np
import networkx as nx
import os
import gc
from common import create_triangle_list
## contact data
all_datasets = ["InVS13", "InVS15", "LH10", "LyonSchool", "Thiers13"]
names = {
    'InVS13': "cont-workplace-13",
    "InVS15": "cont-workplace-15",
    "LH10": "cont-hospital",
    "LyonSchool": "cont-primary-school",
    "Thiers13": "cont-high-school"
}

for dataset in all_datasets:
    print(dataset)
    dir_name = names[dataset]
    os.makedirs(f"../CleanData/{dir_name}", exist_ok=True)
    
    copresence_df = pd.read_csv(f"../RawData/contact-data/co-presence/tij_pres_{dataset}.dat", 
                                sep=" ", names = ["t", "i", "j"])
    labels_df = pd.read_csv(f"../RawData/contact-data/metadata/metadata_{dataset}.dat", 
                            sep="\t", names = ["id", "group"])
    
    all_labels = [ i for i in pd.unique(labels_df['group'])]
    label_dict = {name: i for i, name in enumerate(all_labels)}
    labels_df['group_code'] = labels_df['group'].map(label_dict) 
    labels_df[['id', 'group_code']].to_csv(f'../Data/{dir_name}/labels.csv', index=False)
    
    ct = 0
    elist = []
    tlist = []
    slist = []
    for t, df in copresence_df.groupby("t"):
        
        G = nx.from_pandas_edgelist(df, 'i', 'j')
        
        
        for i,j in G.edges():
            elist.append((min(i,j), max(i,j), t))
        
        curr_tlist = create_triangle_list(G)
        for i,j,k in curr_tlist:
            i,j,k = np.sort([i,j,k])
            tlist.append((i,j,k, t))
        
        ccs = nx.find_cliques(G)
        for cc in ccs:
            slist.append(sorted(list(cc)) + [t])
            
    e_df = pd.DataFrame(elist)
    e_df.columns = ['node_1', 'node_2', 't']
    e_df = e_df[['node_1', 'node_2']].drop_duplicates()

    t_df = pd.DataFrame(tlist)
    t_df.columns = ['node_1', 'node_2', 'node_3', 't']
    t_df = t_df[['node_1', 'node_2', 'node_3']].drop_duplicates()

    e_df.to_csv(f'../Data/{dir_name}/edges.csv', index=False)
    t_df.to_csv(f'../Data/{dir_name}/triangles.csv', index=False)
    with open(f'../Data/{dir_name}/simplices.csv', 'w') as f:
        for item in slist:
            f.write("%s\n" % ",".join([str(i) for i in item]))


