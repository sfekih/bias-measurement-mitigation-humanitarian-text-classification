import os
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from typing import List, Set, Dict
from scipy import stats
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

mappings = {
    "gender": {
        "F<->N": ["female", "neutral"],
        "F<->M": ["female", "male"],
        "M<->N": ["male", "neutral"],
    },
    "country": {
        "C<->V": ["canada", "venezuela"],
        "C<->S": ["canada", "syria"],
        "S<->V": ["syria", "venezuela"],
    },
}

mappings_bidirectional = {
    "gender": {
        "N->F": ["neutral", "female"],
        "N->M": ["neutral", "male"],
        "M->F": ["male", "female"],
        "F->N": ["female", "neutral"],
        "M->N": ["male", "neutral"],
        "F->M": ["female", "male"],
    },
    "country": {
        "C->V": ["canada", "venezuela"],
        "C->S": ["canada", "syria"],
        "S->V": ["syria", "venezuela"],
        "V->C": ["venezuela", "canada"],
        "S->C": ["syria", "canada"],
        "V->S": ["venezuela", "syria"],
    },
}

mapping_training_type_to_name = {
    "no_debiasing": "Base",
    "counterfactual_debiasing": "CDA",
    "no_finetuning": "No-FT",
    "focal_loss_debiasing": "FL",
    "all_debiasing": "CDA+FL",
}

tags = [
    "first_level_tags->pillars_1d->Casualties",
    "first_level_tags->pillars_1d->Context",
    "first_level_tags->pillars_1d->Covid-19",
    "first_level_tags->pillars_1d->Displacement",
    "first_level_tags->pillars_1d->Humanitarian access",
    "first_level_tags->pillars_1d->Information and communication",
    "first_level_tags->pillars_1d->Shock/event",
    "first_level_tags->pillars_2d->At risk",
    "first_level_tags->pillars_2d->Capacities & response",
    "first_level_tags->pillars_2d->Humanitarian conditions",
    "first_level_tags->pillars_2d->Impact",
    "first_level_tags->pillars_2d->Priority interventions",
    "first_level_tags->pillars_2d->Priority needs",
    "first_level_tags->sectors->Agriculture",
    "first_level_tags->sectors->Cross",
    "first_level_tags->sectors->Education",
    "first_level_tags->sectors->Food security",
    "first_level_tags->sectors->Health",
    "first_level_tags->sectors->Livelihoods",
    "first_level_tags->sectors->Logistics",
    "first_level_tags->sectors->Nutrition",
    "first_level_tags->sectors->Protection",
    "first_level_tags->sectors->Shelter",
    "first_level_tags->sectors->Wash",
]

sorted_columns = [
    "Measurement",
    "LLM",
    "Model",
    "F<->M",
    "F<->N",
    "M<->N",
    "G-Avg",
    "C<->V",
    "C<->S",
    "S<->V",
    "C-Avg",
    "T-Avg",
]


# Classification results
classification_tags = {
    "mean->first_level_tags": "Avg",
    "mean->first_level_tags->pillars_1d": "pillars 1d",
    "mean->first_level_tags->pillars_2d": "pillars 2d",
    "mean->first_level_tags->sectors": "sectors",
}
results_metrics = ["precision", "f_score"]

ordered_classification_results = [
    "Measurement",
    "LLM",
    "Model",
    "sectors precision",
    "sectors f_score",
    "pillars 1d precision",
    "pillars 1d f_score",
    "pillars 2d precision",
    "pillars 2d f_score",
    "Avg precision",
    "Avg f_score",
]

label_ticks = {
    "pillars_1d": sorted(
        [
            "Covid-19",
            "Displacement",
            "Context",
            "Casualties",
            "Information and\ncommunication",
            "Humanitarian\naccess",
            "Shock/event",
        ]
    ),
    "pillars_2d": sorted(
        [
            "Impact",
            "Humanitarian\nconditions",
            "Capacities &\nresponse",
            "At risk",
            "Priority\ninterventions",
            "Priority needs",
        ]
    ),
    "sectors": sorted(
        [
            "Logistics",
            "Health",
            "Cross",
            "Education",
            "Shelter",
            "Food security",
            "Protection",
            "Livelihoods",
            "Wash",
            "Agriculture",
            "Nutrition",
        ]
    ),
}


def generate_embedding_results(
    llm_models: List[str], training_setups: List[str], folder_name: str
):
    final_results_df = pd.DataFrame()

    with tqdm(total=len(llm_models) * len(training_setups)) as pbar:
        for one_llm in llm_models:
            for one_training_setup in training_setups:
                distances = {}
                for one_protected_attribute in ["gender", "country"]:
                    df = pd.read_csv(
                        os.path.join(
                            folder_name,
                            one_llm,
                            one_training_setup,
                            "embeddings",
                            f"{one_llm}_{one_training_setup}_base_architecture_{one_protected_attribute}_embeddings.csv",
                        )
                    )
                    df["kword_type"] = df["kword_type"].apply(
                        lambda x: literal_eval(x)[0]
                    )
                    df["embeddings"] = df["embeddings"].apply(literal_eval)
                    n_ids = df.entry_id.nunique()

                    shifts = defaultdict(list)

                    keywords_one_protected_attribute = mappings[one_protected_attribute]

                    for i in range(0, len(df), 3):
                        df_one_entry = df.iloc[i : i + 3]
                        assert (
                            df_one_entry.entry_id.nunique() == 1
                        ), "issue with df entry id"
                        for name, kwords in keywords_one_protected_attribute.items():
                            base_kword = kwords[0]
                            counterfactual_kword = kwords[1]

                            base_output = df_one_entry[
                                df_one_entry.kword_type == base_kword
                            ].embeddings.values[0]
                            counterfactual_output = df_one_entry[
                                df_one_entry.kword_type == counterfactual_kword
                            ].embeddings.values[0]
                            dist = euclidean_distances(
                                [counterfactual_output], [base_output]
                            )[0][0]

                            shifts[name].append(np.abs(dist))

                    for name, tagwise_distances in shifts.items():
                        assert len(tagwise_distances) == n_ids

                    shifts = {
                        name: np.median(tagwise_distances)  # / n_ids
                        for name, tagwise_distances in shifts.items()
                    }
                    shifts[f"{one_protected_attribute[0].capitalize()}-Avg"] = np.mean(
                        list(shifts.values())
                    )
                    distances.update(shifts)

                final_results_df = final_results_df.append(
                    dict(
                        list(
                            {
                                "Measurement": "Embeddings",
                                "LLM": one_llm,
                                "Model": mapping_training_type_to_name[
                                    one_training_setup
                                ],
                            }.items()
                        )
                        + list(distances.items())
                        + list({"T-Avg": np.mean(list(distances.values()))}.items())
                    ),
                    ignore_index=True,
                )
                pbar.update(1)

    final_results_df = final_results_df[sorted_columns]
    return final_results_df


def get_classification_results(llm_models, training_setups, folder_name: str):
    final_results_df = pd.DataFrame()

    for one_llm in llm_models:
        for one_training_setup in training_setups:
            metrics_one_row = {
                "Measurement": "Classification Results",
                "LLM": one_llm,
                "Model": mapping_training_type_to_name[one_training_setup],
            }
            df = pd.read_csv(
                os.path.join(
                    folder_name,
                    one_llm,
                    one_training_setup,
                    "classification",
                    f"classification_results_test_df_{one_llm}_{one_training_setup}_base_architecture.csv",
                )
            ).drop(columns=["Unnamed: 0"])

            for one_results_col, procesed_results_col in classification_tags.items():
                metrics_one_tag = df[df.tag == one_results_col]
                for one_metic in ["precision", "f_score"]:
                    metrics_one_row[
                        f"{procesed_results_col} {one_metic}"
                    ] = metrics_one_tag[one_metic].iloc[0]

            final_results_df = final_results_df.append(
                metrics_one_row, ignore_index=True
            )

    final_results_df = final_results_df[ordered_classification_results]

    return final_results_df


def _generate_heatmaps(
    df: pd.DataFrame,
    fig_name: str = None,
    heatmap_col: str = "median_shift",
    multiplication_factor: int = 1,
):
    treated_level0_tags = ["sectors", "pillars_1d", "pillars_2d"]
    processed_df = df.copy().sort_values(by=["tag", "original->couterfactual"])
    base_kwords = processed_df.base_kw.unique().tolist()
    n_base_kwords = len(base_kwords)

    processed_df = processed_df[
        processed_df.tag.apply(
            lambda x: any(
                [f"first_level_tags->{one_tag}" in x for one_tag in treated_level0_tags]
            )
        )
    ]
    processed_df[heatmap_col] = (
        multiplication_factor * processed_df[heatmap_col]
    ).round(3)

    min_val = min(processed_df[heatmap_col].min(), -1e-7)
    max_val = processed_df[heatmap_col].max()
    norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

    f, axes = plt.subplots(
        n_base_kwords,
        len(treated_level0_tags),
        sharey=False,
        sharex=False,
        figsize=(20, 6),
        gridspec_kw={"width_ratios": [1.4, 0.9, 0.8]},
    )

    for i, one_base_kw in enumerate(base_kwords):
        one_kw_df = processed_df[processed_df.base_kw == one_base_kw]

        one_kw_df = one_kw_df.pivot(
            index="tag", columns="original->couterfactual", values=heatmap_col
        )
        for j, one_tag in enumerate(treated_level0_tags):
            one_tag_results = (
                one_kw_df[
                    one_kw_df.index.to_series().apply(
                        lambda x: f"first_level_tags->{one_tag}" in x
                    )
                ]
                .copy()
                .T
            )

            xticks = label_ticks[one_tag]
            # print(xticks)
            cbar_ax = f.add_axes([0.91, 0.15, 0.03, 0.7])
            g = sns.heatmap(
                one_tag_results,
                cmap="coolwarm",
                # cbar=True if i == len(treated_level0_tags) - 1 else False,
                cbar_ax=cbar_ax,
                ax=axes[i, j],
                # vmin=-max_abs if min_val<0 else 0,
                vmin=min_val,
                vmax=max_val,
                norm=norm,
                annot=True,
                xticklabels=xticks,
                # linewidths=0.1,
                annot_kws={"fontsize": 10},
            )
            if j != 0:
                g.get_yaxis().set_visible(False)
            if i != 2:
                g.get_xaxis().set_visible(False)

            g.set_ylabel("")
            g.tick_params(labelsize=15)
            if i == 2:
                g.set_xlabel(one_tag, fontsize=18)
            else:
                g.set_xlabel("")

            plt.yticks(rotation=0)

    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")


def generate_probability_tag_results(
    llm_model: str,
    training_setup: str,
    folder_name: str,
    one_protected_attribute: str,
):
    df = pd.read_csv(
        os.path.join(
            folder_name,
            llm_model,
            training_setup,
            "predictions_discrepency",
            f"{llm_model}_{training_setup}_base_architecture_{one_protected_attribute}_predictions_discrepency.csv",
        )
    )

    df["kword_type"] = df["kword_type"].apply(lambda x: literal_eval(x)[0])

    keywords_one_protected_attribute = mappings[one_protected_attribute]

    shifts = defaultdict(lambda: defaultdict(list))

    for i in range(0, len(df), 3):
        df_one_entry = df.iloc[i : i + 3]
        assert df_one_entry.entry_id.nunique() == 1, "issue with df entry id"

        if not df_one_entry.isnull().values.any():
            for (
                name,
                kwords,
            ) in keywords_one_protected_attribute.items():
                base_kword = kwords[0]
                counterfactual_kword = kwords[1]

                base_df = df_one_entry[df_one_entry.kword_type == base_kword]
                counterfactual_df = df_one_entry[
                    df_one_entry.kword_type == counterfactual_kword
                ]
                for one_tag in tags:
                    dist = 100 * (
                        counterfactual_df[f"probability_{one_tag}"].values[0]
                        - base_df[f"probability_{one_tag}"].values[0]
                    )

                    shifts[name][one_tag].append(dist)

    shifts_df = pd.DataFrame()
    for abbrev_name, tagwise_distances in shifts.items():
        real_names = [
            one_real_name.capitalize()
            for one_real_name in mappings[one_protected_attribute][abbrev_name]
        ]

        for tag, all_distances_one_tag in tagwise_distances.items():
            median_distance = np.median(all_distances_one_tag)
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[0],
                    "original->couterfactual": "->".join(real_names),
                    "median_shift": median_distance,
                },
                ignore_index=True,
            )
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[1],
                    "original->couterfactual": "->".join(real_names[::-1]),
                    "median_shift": -median_distance,
                },
                ignore_index=True,
            )

    return shifts_df


def generate_explainability_tag_results(
    llm_model: str,
    training_setup: str,
    folder_name: str,
    one_protected_attribute: str,
    explainability_method: str = "Layer DeepLift",
):
    df = pd.read_csv(
        os.path.join(
            folder_name,
            llm_model,
            training_setup,
            "explainability",
            f"{llm_model}_{training_setup}_base_architecture_{one_protected_attribute}_{explainability_method}_explainability.csv",
        )
    )

    df["kword_type"] = df["kword_type"].apply(lambda x: literal_eval(x)[0])

    keywords_one_protected_attribute = mappings[one_protected_attribute]

    shifts = defaultdict(lambda: defaultdict(list))

    for i in range(0, len(df), 3):
        df_one_entry = df.iloc[i : i + 3]
        assert df_one_entry.entry_id.nunique() == 1, "issue with df entry id"

        if not df_one_entry.isnull().values.any():
            for (
                name,
                kwords,
            ) in keywords_one_protected_attribute.items():
                base_kword = kwords[0]
                counterfactual_kword = kwords[1]

                base_df = df_one_entry[df_one_entry.kword_type == base_kword]
                counterfactual_df = df_one_entry[
                    df_one_entry.kword_type == counterfactual_kword
                ]
                for one_tag in tags:
                    dist = (
                        counterfactual_df[f"explainability_{one_tag}"].values[0]
                        - base_df[f"explainability_{one_tag}"].values[0]
                    )

                    shifts[name][one_tag].append(dist)

    shifts_df = pd.DataFrame()
    for abbrev_name, tagwise_distances in shifts.items():
        real_names = [
            one_real_name.capitalize()
            for one_real_name in mappings[one_protected_attribute][abbrev_name]
        ]

        for tag, all_distances_one_tag in tagwise_distances.items():
            median_distance = np.median(all_distances_one_tag)
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[0],
                    "original->couterfactual": "->".join(real_names),
                    "median_shift": median_distance,
                },
                ignore_index=True,
            )
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[1],
                    "original->couterfactual": "->".join(real_names[::-1]),
                    "median_shift": -median_distance,
                },
                ignore_index=True,
            )

    return shifts_df


def generate_shift_count_tag_results(
    llm_model: str,
    training_setup: str,
    folder_name: str,
    one_protected_attribute: str,
):
    df = pd.read_csv(
        os.path.join(
            folder_name,
            llm_model,
            training_setup,
            "predictions_discrepency",
            f"{llm_model}_{training_setup}_base_architecture_{one_protected_attribute}_predictions_discrepency.csv",
        )
    )

    with open(
        os.path.join(
            folder_name,
            llm_model,
            training_setup,
            "classification",
            f"threshold_values_{llm_model}_{training_setup}_base_architecture.json",
        ),
        "r",
    ) as f:
        thresholds = json.load(f)

    df["kword_type"] = df["kword_type"].apply(lambda x: literal_eval(x)[0])
    n_ids = df.entry_id.nunique()

    keywords_one_protected_attribute = mappings_bidirectional[one_protected_attribute]

    shifts = {
        name: {one_tag: 0 for one_tag in tags}
        for name in keywords_one_protected_attribute.keys()
    }

    for i in range(0, len(df), 3):
        df_one_entry = df.iloc[i : i + 3]
        assert df_one_entry.entry_id.nunique() == 1, "issue with df entry id"

        if not df_one_entry.isnull().values.any():
            for (
                name,
                kwords,
            ) in keywords_one_protected_attribute.items():
                base_kword = kwords[0]
                counterfactual_kword = kwords[1]

                base_df = df_one_entry[df_one_entry.kword_type == base_kword]
                counterfactual_df = df_one_entry[
                    df_one_entry.kword_type == counterfactual_kword
                ]
                for one_tag in tags:
                    base_probability = base_df[f"probability_{one_tag}"].values[0]

                    counterfactual_probability = counterfactual_df[
                        f"probability_{one_tag}"
                    ].values[0]

                    tag_threshold = thresholds[one_tag.replace("probability_", "")]

                    base_ratio = base_probability / tag_threshold
                    counterfactual_ratio = counterfactual_probability / tag_threshold

                    if (base_ratio < 1) and (counterfactual_ratio >= 1):
                        shifts[name][one_tag] += 1

    shifts_df = pd.DataFrame()
    for abbrev_name, tagwise_distances in shifts.items():
        real_names = [
            one_real_name.capitalize()
            for one_real_name in mappings_bidirectional[one_protected_attribute][
                abbrev_name
            ]
        ]

        for tag, all_distances_one_tag in tagwise_distances.items():
            shifts_df = shifts_df.append(
                {
                    "tag": tag,
                    "base_kw": real_names[0],
                    "original->couterfactual": "->".join(real_names),
                    "median_shift": all_distances_one_tag / n_ids,
                },
                ignore_index=True,
            )

    return shifts_df


all_visualized_methods = {
    "probability_discrepancy": generate_probability_tag_results,
    "explainability_discrepancy": generate_explainability_tag_results,
    "shifts_count": generate_shift_count_tag_results,
}


def generate_tagwise_results(
    llm_models: List[str],
    training_setups: List[str],
    protected_attributes: List[str],
    visualized_methods: List[str],
    folder_name: str,
    generate_vizus: bool = True,
):
    all_results_df = pd.DataFrame()
    with tqdm(
        total=len(llm_models)
        * len(training_setups)
        * len(protected_attributes)
        * len(visualized_methods)
    ) as pbar:
        for llm_model in llm_models:
            for training_setup in training_setups:
                for method in visualized_methods:
                    results_generation_function = all_visualized_methods[method]
                    for protected_attr in protected_attributes:
                        visu_file = os.path.join(
                            folder_name,
                            "visualizations",
                            llm_model,
                            training_setup,
                            method,
                            protected_attr,
                        )
                        os.makedirs(visu_file, exist_ok=True)
                        results_df_one_method = results_generation_function(
                            llm_model=llm_model,
                            training_setup=training_setup,
                            folder_name=folder_name,
                            one_protected_attribute=protected_attr,
                        )
                        results_df_one_method["llm_model"] = llm_model
                        results_df_one_method["training_setup"] = training_setup
                        results_df_one_method["method"] = method
                        results_df_one_method["protected_attr"] = protected_attr
                        all_results_df = pd.concat(
                            [all_results_df, results_df_one_method]
                        )

                        if generate_vizus:
                            figname = os.path.join(
                                visu_file,
                                f"{llm_model}_{training_setup}_{method}_{protected_attr}.png",
                            )
                            _generate_heatmaps(
                                results_df_one_method,
                                figname,
                            )

                        pbar.update(1)
    return all_results_df
