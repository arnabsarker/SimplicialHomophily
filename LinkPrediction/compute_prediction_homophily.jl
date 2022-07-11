include("common.jl")
include("open_triangle_prediction.jl")

using ScHoLP
using CSV
using StatsBase
using Random
Random.seed!(11);

const start_split = 25;
const mid_split = 50;
const end_split = 100;


"""
homophily_score
---------------
Returns the arithmetic mean of the weights of the edges of a list of triangles.
arithmetic_mean(triangles::Vector{NTuple{3,Int64}}, B::SpIntMat)
Input parameters:
- triangles::Vector{NTuple{3,Int64}}: The vector of triangles upon which to compute scores.
- B::SpIntMat: Projected graph as a Sparse integer matrix, where B[i, j] is the number of times that i and j co-appear in a simplex.
"""
function homophily_score(triangles::Vector{NTuple{3,Int64}}, L::Dict)
    scores = zeros(Float64, length(triangles))
    Threads.@threads for ind = 1:length(triangles)
        i, j, k = triangles[ind]
        scores[ind] = (L[i] == L[j]) * (L[j] == L[k]) * (L[i] == L[k]) + 0
    end
    return scores
end

function collect_homophily_scores(dataset::HONData)
    triangles = read_data(dataset, mid_split, end_split)[1]
    L = CSV.File("data/$(dataset.name)/$(dataset.name)-labels.csv") |> Dict;
    write_scores(dataset, "homophily", homophily_score(triangles, L))
end

function collect_logreg_homophily_supervised_scores(dataset::HONData)
    function feature_matrix(triangles::Vector{NTuple{3,Int64}},
                            At::SpIntMat, B::SpIntMat, L::Dict)
        degrees = vec(sum(make_sparse_ones(B), dims=1))
        simp_degrees = vec(sum(At, dims=1))
        common_nbrs = common_neighbors_map(B, triangles)
        ntriangles = length(triangles)
        X = zeros(Float64, 27, ntriangles)
        Threads.@threads for ind = 1:ntriangles
            i, j, k = triangles[ind]
            X[1:3, ind] = [B[i, j]; B[j, k]; B[i, k]]
            X[4:6, ind] = degrees[[i, j, k]]
            X[7:9, ind] = simp_degrees[[i, j, k]]
            common_ij = common_nbr_set(common_nbrs, i, j)
            common_ik = common_nbr_set(common_nbrs, i, k)
            common_jk = common_nbr_set(common_nbrs, j, k)
            X[10, ind] = length(common_ij)
            X[11, ind] = length(common_ik)
            X[12, ind] = length(common_jk)
            X[13, ind] = length(intersect(common_ij, common_ik, common_jk))
            X[14:22, ind] = log.(X[1:9, ind])
            X[23:26, ind] = log.(X[10:13, ind] .+ 1.0)
            X[27, ind] = (L[i] == L[j]) + (L[j] == L[k]) + (L[i] == L[k])
        end
        return Matrix(X')
    end
    
    triangles = read_data(dataset, mid_split, end_split)[1]
    simplices = dataset.simplices
    nverts = dataset.nverts
    times = dataset.times
    old_simplices, old_nverts = split_data(simplices, nverts, times, mid_split, end_split)[1:2]
    A, At, B = basic_matrices(old_simplices, old_nverts)
    L = CSV.File("data/$(dataset.name)/$(dataset.name)-labels.csv") |> Dict;
    basename = basename_str(dataset.name)

    train_triangles, val_labels = read_data(dataset, start_split, mid_split)
    train_simplices, train_nverts = split_data(simplices, nverts, times, start_split, mid_split)[1:2]
    At_train, B_train = basic_matrices(train_simplices, train_nverts)[2:3]
    X_train = feature_matrix(train_triangles, At_train, B_train, L)
    dt = fit(ZScoreTransform, X_train, dims=1)
    X_train = StatsBase.transform(dt, X_train)
    model = LogisticRegression(fit_intercept=true, solver="liblinear")
    ScikitLearn.fit!(model, X_train, val_labels)
    X = feature_matrix(triangles, At, B, L)
    X = StatsBase.transform(dt, X)
    learned_scores = ScikitLearn.predict_proba(model, X)[:, 2]
    write_scores(dataset, "logreg_homophily_supervised", learned_scores)
end;


function main(dataset_name)
    try
        dataset = read_txt_data(dataset_name)
        collect_labeled_dataset(dataset)
        collect_local_scores(dataset)
        collect_walk_scores(dataset)
        collect_logreg_supervised_scores(dataset)
        collect_homophily_scores(dataset)
        collect_logreg_homophily_supervised_scores(dataset)
        collect_Simplicial_PPR_decomposed_scores(dataset)
    catch e
        println("error for $(dataset_name)")
        println(e.msg)
    end
end


# all_datasets = ["cont-hospital", "cont-workplace-13", "cont-workplace-15", "cont-village", 
#                 "bills-senate", "bills-house", "hosp-DAWN", "contact-primary-school", "contact-high-school"]
datasets_dict = Dict(1 => ["cont-hospital", "cont-workplace-15"], 
    2 => ["cont-workplace-13", "coauth-dblp"], 
    3 => ["cont-village"], 
    4 => ["bills-senate"],
    5 => ["bills-house"], 
    6 => ["hosp-DAWN"],
    7 => ["cont-high-school"],
    8 => ["cont-primary-school"])

datasets_dict = Dict(1 => [], 
    2 => ["coauth-dblp"], 
    3 => [], 
    4 => [],
    5 => [], 
    6 => [],
    7 => [],
    8 => [])
    
my_datasets = datasets_dict[parse(Int64, string(ARGS[1]))]

for d in my_datasets
    main(d)
end