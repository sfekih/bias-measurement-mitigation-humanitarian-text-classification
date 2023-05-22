from debiasing_pipelines.counterfactual_explainability_discrepency.ModelsExplainability import (
    MultiLabelClassificationExplainer,
)
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import List, Set, Dict
from IPython.display import display
from debiasing_pipelines.utils import _get_tags_based_stats
from utils import _flatten


def _multisets_intersection(set_list: List[Set]):
    return list(set.intersection(*set_list))


def _get_explainability_results_one_id(cls_explainer, excerpts: List[str]):
    """
    {tag: []}
    """
    # raw_explainability_results: {tag: [{token1: explainability score, ...}, {...}, ...]}
    raw_explainability_results = defaultdict(list)
    for one_excerpt in excerpts:
        results_one_excerpt = cls_explainer(one_excerpt)
        for tag, explanability_per_tag_excerpt in results_one_excerpt.items():
            raw_explainability_results[tag].append(explanability_per_tag_excerpt)

    all_tokens = list(raw_explainability_results.values())[0]
    # all tokens: List[List[token]]
    all_tokens = [
        set(list(one_excerpt_results.keys())) for one_excerpt_results in all_tokens
    ]

    tokens_intersection = _multisets_intersection(all_tokens)

    # cleaned_explainability_results: {tag: [explainability score excerpt 1, ...], ...}
    cleaned_explainability_results = {
        one_tag: [
            np.mean(
                [
                    explainability_score_one_token
                    for token, explainability_score_one_token in one_excerpt_results.items()
                    if token not in tokens_intersection
                ]
            )
            for one_excerpt_results in explainability_results_one_tag
        ]
        for one_tag, explainability_results_one_tag in raw_explainability_results.items()
    }

    return cleaned_explainability_results


# def _keep_relevant_vals(
#     all_embeddings: Dict[str, float], intersection_tokens: List[str]
# ):
#     return [
#         val for key, val in all_embeddings.items() if key not in intersection_tokens
#     ]


def get_explaiability_results(model, data: pd.DataFrame, attribution_type: str, device):
    results_df = data.copy().sort_values(by="entry_id")
    cls_explainer = MultiLabelClassificationExplainer(model, attribution_type, device)

    explainability_results_df = pd.DataFrame()

    for id in tqdm(results_df.entry_id.unique().tolist()):
        df_one_id = results_df[results_df.entry_id == id].copy()
        explainability_results_one_id = _get_explainability_results_one_id(
            cls_explainer, df_one_id.excerpt.tolist()
        )
        # append results to df
        for tag, results_one_tag in explainability_results_one_id.items():
            df_one_id[f"explainability_{tag}"] = results_one_tag

        explainability_results_df = pd.concat([explainability_results_df, df_one_id])

    return explainability_results_df
