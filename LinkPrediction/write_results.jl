include("common.jl")
include("open_triangle_prediction.jl")

function main()
    all_datasets = ["cont-hospital", "cont-workplace-13", "cont-workplace-15", "hosp-DAWN", "bills-senate", 
                    "bills-house", "coauth-dblp", "cont-primary-school", "cont-high-school"]
    # all score types
#     score_types = ["harm_mean", "geom_mean", "arith_mean", "common", "jaccard", "adamic_adar", "proj_graph_PA", 
#         "simplex_PA", "UPKatz", "WPKatz", "UPPR", "WPPR", "homophily", "logreg_supervised", "logreg_homophily_supervised"]
    score_types = ["homophily", "logreg_supervised", "logreg_homophily_supervised"]
    
    open("prediction-scores/results_50_recreation.txt", "w") do io
        write(io, "Dataset,Metric,Value\n")
        for dataset_name=all_datasets
            dataset = read_txt_data(dataset_name)
            triangles, labels = read_data(dataset, 50, 100)
            rand_rate = sum(labels .== 1) / length(labels)
            write(io, "$(dataset_name),random,$(rand_rate)\n")
            for score_type in score_types
                try
                    scores = read_scores(dataset, score_type)
                    ave_prec = average_precision_score(labels, scores)
                    improvement = ave_prec / rand_rate
                    write(io, "$(dataset_name),$(score_type),$(improvement)\n")
                catch
                    println("Did not work for $(dataset_name),$(score_type)")
                end
            end
        end
    end
end;

main()