from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from collections import defaultdict
from debiasing_pipelines.utils import _get_stats_one_tag


def get_embeddings_results(trained_model, data):
    """
    output: Dict[tag, pd.DataFrame[entry_id, excerpt, one column per keyword containing probabilities]]
    """

    results_df = data.copy()
    results_df.sort_values(by="entry_id", inplace=True)

    embeddings = trained_model.custom_predict(results_df, setup="model_embeddings")
    embeddings = [
        embedding_one_excerpt.numpy().round(4).tolist()
        for embedding_one_excerpt in embeddings
    ]

    results_df["embeddings"] = embeddings

    return results_df

    # per_tag_raw_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # unique_ids = data["entry_id"].unique().tolist()

    # for one_id in unique_ids:
    #     df_one_id = results_df[results_df.entry_id == one_id]
    #     row_original = df_one_id[df_one_id.excerpt_type == "original"]
    #     kword_type_original = row_original.kword_type.values[0][0]
    #     rows_counterfactual = df_one_id[df_one_id.excerpt_type != "original"]

    #     original_embedding = row_original[f"embeddings"].values[0]
    #     for i, row_counterfactual in rows_counterfactual.iterrows():
    #         one_counterfactual_embedding = row_counterfactual[f"embeddings"]
    #         one_counterfactual_kword_type = row_counterfactual.kword_type[0]

    #         per_tag_raw_results["euclidean_distances"][kword_type_original][
    #             one_counterfactual_kword_type
    #         ].append(
    #             euclidean_distances([original_embedding, one_counterfactual_embedding])
    #         )

    #         per_tag_raw_results["cosine_distances"][kword_type_original][
    #             one_counterfactual_kword_type
    #         ].append(
    #             cosine_distances([original_embedding, one_counterfactual_embedding])
    #         )

    # total_stats = {
    #     one_distance_type: _get_stats_one_tag(one_distance_list)
    #     for one_distance_type, one_distance_list in per_tag_raw_results.items()
    # }

    # return total_stats
