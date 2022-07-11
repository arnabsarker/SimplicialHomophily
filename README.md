# Simplicial Homophily

Code and available data for the work 
* Generalizing Homophily to Simplicial Complexes. Arnab Sarker, Natalie Northrup, Ali Jadbabaie. <em>Complex Networks</em> (2022).

## Dataset Descriptions
The data used in this work falls broadly into six categories depending on the meaning of the interactions.

-   `cont` indicates relationships are due to physical contact
    interactions based on bluetooth proximity. Contacts are measured in
    20 second intervals, creating a graph of contacts each 20 second
    period, from which we create simplices from maximal cliques of this
    graph. Node labels are determined by membership in a classroom
    (`cont-primary-school`, `cont-high-school`), specific occupation
    within a hospital (`cont-hospital`), household in a village
    (`cont-village`), or department within a workplace
    (`cont-workplace-13`, `cont-workplace-15`). These datasets are available publicly [here](http://www.sociopatterns.org/datasets/) (Génois and Barrat, 2018).

-   `bills` details connections via cosponsorship in the United States
    federal government. Simplices form when members of congress
    cosponsor bills with one another, and nodes are labeled by party
    affiliation (`bills-house`, `bills-senate`). These datasets are available publicly [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMZS8E) (Fowler, 2006).

-   `email` corresponds to social connections created via email.
    Simplices form when Enron employees send emails to groups of others,
    and labels are based on gender of associated nodes (`email-Enron`). This dataset is available publicly [here](https://www.cs.cmu.edu/~enron/) (Benson et al., 2018).

-   `hosp` refers to connections due to multiple drugs being used by the
    same patient in an emergency room. Nodes are drugs, and simplices
    form when an individual enters the emergency room having used
    multiple drugs. Drugs are labeled based on drug classifications from
    the Drug Abuse Warning Network (`hosp-DAWN`). This dataset is publicly available [here](https://www.datafiles.samhsa.gov/dataset/drug-abuse-warning-network-2011-dawn-2011-ds0001) (Benson et al., 2018).

-   `soc` refers to social relationships in the context of an online
    social network. Nodes are individuals, and edges represent
    interactions within the context of the network. Each node is also
    able to be a member of any number of groups in the social network.
    2-simplices form when three nodes all share edges with one another
    and there is some group that connects the three nodes. The label of
    a node is given by the group label for which the node has the most
    mutual edges which are also in that group. For these datasets, due
    to data size restrictions we split the datasets into subsets of
    roughly 20,000 nodes. To ensure sufficiently many 2-simplices in
    each subset of 5,000 nodes, we restrict the data to only use nodes
    that are in the the 10 largest groups for `soc-youtube`,
    `soc-flickr`, and `soc-livejorunal`, and `soc-orkut`. These datasets are publicly available [here](https://socialnetworks.mpi-sws.org/data-imc2007.html) (Mislove et al., 2007).

-   `retail` refers to relationships between hotels in a retail dataset.
    Simplices form when multiple hotels which are browsed during the
    same user session on the website Trivago. Node labels correspond to
    the country where the hotel is located. The dataset is publicly available [here](https://www.cs.cornell.edu/~arb/data/) (Benson et al., 2018).

## Figure Replication

The code is divided into three folders:

[EmpiricalExperiments/](EmpiricalExperiments/) (Table 1, Figure 3, Figure 4)

[SyntheticExperiments/](SyntheticExperiments/) (Figure 2)

[LinkPrediction/](LinkPrediction) (Table 2)

---

With the exception of Table 1, all figures can be recreated using the linked notebooks.


[EmpiricalExperiments/](EmpiricalExperiments/)

* [Table 1](EmpiricalExperiments/Data/Table1_DataSummary.ipynb)

    The cleaned data is present for available datasets in [EmpiricalExperiments/Data/](EmpiricalExperiments/Data/).
Code for cleaning the data can be found under [EmpiricalExperiments/DataCleaningCode/](EmpiricalExperiments/DataCleaningCode/), 
but not all raw data is available as some datasets are large. In such cases, links to raw data is available.
The code to generate Table 1 can be found [here](EmpiricalExperiments/Data/Table1_DataSummary.ipynb).

* [Figure 3](EmpiricalExperiments/Figures/Fig3_and_ExplainedVarianceAnalysis_GlobalResults.ipynb)

    Code to generate global homophily scores (and bootstrapped values for confidence intervals)
can be found in [EmpiricalExperiments/Code/global_homophily_comp.py](EmpiricalExperiments/Code/global_homophily_comp.py) (and [global_bootstrapping.py](EmpiricalExperiments/Code/global_bootstrapping.py)).

    All homophily scores are precomputed and available in this supplementary material.
The code for the actual figure is in this [notebook](EmpiricalExperiments/Figures/Fig3_and_ExplainedVarianceAnalysis_GlobalResults.ipynb), which also contains code for establishing that edge homophily explains nearly 70% of variance
in hypergraph homophily.

* [Figure 4](EmpiricalExperiments/Figures/Fig4_ClassBasedFigures.ipynb)

    Heterogeneous homophily scores and bootstrapping are computed in 
[homophily_numbered.py](EmpiricalExperiments/Code/homophily_numbered.py) and [homophily_bootstrapping.py](EmpiricalExperiments/Code/homophily_bootstrapping.py), respectively. 
As with Figure 3, we precompute all values and report them in a single [notebook](EmpiricalExperiments/Figures/Fig4_ClassBasedFigures.ipynb).


[SyntheticExperiments/](SyntheticExperiments/)

* [Figure 2](SyntheticExperiments/Fig2_AnalyzeSSBM.ipynb)

    Code to generate the synthetic data is in [SyntheticExperiments/GenerateSyntheticData.ipynb](SyntheticExperiments/GenerateSyntheticData.ipynb).
Then, using [EmpiricalExperiments/Code/global_homophily_comp.py](EmpiricalExperiments/Code/global_homophily_comp.py), we compute the
homogenous homophily score of the synthetic datasets and store them in [SyntheticExperiments/Results](SyntheticExperiments/Results)
The resulting information is then used in [SyntheticExperiments/Fig2_AnalyzeSSBM.ipynb](SyntheticExperiments/Fig2_AnalyzeSSBM.ipynb)
to generate the plots of Figure 2.


[LinkPrediction/](LinkPrediction) 

* [Table 2](LinkPrediction/Table2_LinkPredictionResults.ipynb)
    
    The homophily scores in Table 2 are computed [here](LinkPrediction/Data-50/Homophily50.ipynb),
as the directory contains the first 50% of data in each dataset.
    Since not all data can be submitted with the supplement, we retain the file [homophily_comps.csv](LinkPrediction/Data-50/homophily_comps.csv), which contains all needed homophily scores for Table 2.
To get the data in the correct form to apply the Julia code, see [Reformat-ScHoLP.ipynb](EmpiricalExperiments/DataProcessingCode/Reformat-ScHoLP.ipynb)
The logistic regression experiments are computed using [compute_prediction_homophily.jl](LinkPrediction/compute_prediction_homophily.jl).
Due to size of large datasets, in this repository we retain results for small datasets.


## References

Benson, Austin R., et al. "Simplicial closure and higher-order link prediction." Proceedings of the National Academy of Sciences 115.48 (2018): E11221-E11230.

Fowler, James H. "Connecting the Congress: A study of cosponsorship networks." Political Analysis 14.4 (2006): 456-487.

Génois, Mathieu, and Barrat, Alain. "Can co-location be used as a proxy for face-to-face contacts?." EPJ Data Science 7.1 (2018): 1-18.

Mislove, Alan, et al. "Measurement and analysis of online social networks." Proceedings of the 7th ACM SIGCOMM conference on Internet measurement. 2007.