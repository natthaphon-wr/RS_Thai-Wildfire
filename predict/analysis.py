import argparse
import logging
import warnings
import os
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_stats(df):
    total_pixels = 512*512
    df["ratio_burned"] = df["burned"] / total_pixels
    df["ratio_cloud"] = df["cloud"] / total_pixels
    df["ratio_burned_on_clear"] = (df["burned_clear"]/df["clear"]).fillna(0)
    df["ratio_burned_on_cloud"] = (df["burned_cloud"]/df["cloud"]).fillna(0)
    stats_df = df[["filename", "ratio_cloud", "ratio_burned", "ratio_burned_on_clear", "ratio_burned_on_cloud"]]
    return stats_df

def sum_res(df):
    cols = ["ratio_burned", "ratio_cloud", "ratio_burned_on_clear", "ratio_burned_on_cloud"]
    summary_df = pd.DataFrame({
        "mean": df[cols].mean(),
        "median": df[cols].median(),
        "min": df[cols].min(),
        "max": df[cols].max(),
        "std": df[cols].std()
    })
    summary_df = summary_df.reset_index().rename(columns={"index": "measures"})
    return summary_df

def compare_visual(df):
    def create_boxplot(data_pos, data_neg, title):
        fig, ax = plt.subplots()
        ax.boxplot([data_pos, data_neg], labels=["Positive", "Negative"])
        ax.set_ylabel("Burned Ratio")
        ax.set_title(title)
        plt.close(fig)
        return fig

    # 1. Boxplot all region
    fig_all = create_boxplot(data_pos = df["pos_ratio_burned"], 
                             data_neg = df["neg_ratio_burned"], 
                             title = "All Region")

    # 2. Boxplot on clear region
    fig_clear = create_boxplot(data_pos = df["pos_ratio_burned_on_clear"], 
                               data_neg = df["neg_ratio_burned_on_clear"], 
                               title = "Clear Region")

    # 3. Boxplot on cloud region
    fig_cloud = create_boxplot(data_pos = df["pos_ratio_burned_on_cloud"],
                               data_neg = df["neg_ratio_burned_on_cloud"],
                               title = "Cloud Region")

    # 4. Scatter burned ratio vs cloud ratio
    fig_scatter, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
    ax[0].scatter(x=df["pos_ratio_cloud"], y=df["pos_ratio_burned"])
    ax[0].set_title("Positive")
    ax[1].scatter(x=df["neg_ratio_cloud"], y=df["neg_ratio_burned"])
    ax[1].set_title("Negative")
    for ax in ax:
        ax.set_xlabel("Cloud Ratio")
        ax.set_ylabel("Burn Ratio")
    plt.suptitle("Scatter plot between cloud ratio and burned ratio")
    plt.tight_layout()
    plt.close(fig_scatter)

    return fig_all, fig_clear, fig_cloud, fig_scatter

def plot_diff_hist(data, title):
    fig, ax = plt.subplots()
    ax.hist(data, bins=20, alpha=0.7)
    ax.axvline(np.median(data), color='red', linestyle='--', label='Median')
    ax.set_title(title)
    ax.legend()
    plt.close()
    return fig

def sign_test(diff_data):
    n_pos = np.sum(diff_data > 0)
    n_neg = np.sum(diff_data < 0)
    n = n_pos + n_neg 
    result_test = stats.binomtest(k=n_pos, n=n, p=0.5, alternative='greater')
    return n_pos, n_neg, result_test.pvalue

def permutation_test(pos_data, neg_data):
    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    
    result = stats.permutation_test(data = (pos_data, neg_data),
                                    statistic=statistic,
                                    permutation_type="samples",
                                    n_resamples=10000,
                                    alternative="greater")
    return result

def bootstrap(diff_data):
    result = stats.bootstrap(
        data = (diff_data,),
        statistic = np.median,  
        vectorized = False,        
        n_resamples = 10000,   
        confidence_level = 0.95,
        method = "percentile",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Analysis of HLS Thai prediction")
    parser.add_argument("--prediction_path", type=str, help="Output path of prediction result")
    parser.add_argument("--data_path", type=str, help="Data path of the prediction")

    args = parser.parse_args()
    prediction_path = args.prediction_path
    DATA_PATH = args.data_path
    
    PREDICTION_POS_PATH = os.path.join(prediction_path, "positive")
    PREDICTION_NEG_PATH = os.path.join(prediction_path, "negative")
    PREDICTION_ANALYSE_PATH = os.path.join(prediction_path, "analysis")
    os.makedirs(PREDICTION_ANALYSE_PATH, exist_ok=True)

    # Read csv result
    res_pos = pd.read_csv(os.path.join(PREDICTION_POS_PATH, "results.csv"))
    res_neg = pd.read_csv(os.path.join(PREDICTION_NEG_PATH, "results.csv"))
    res_pos.drop(columns=["shadow", "obscured", "burned_shadow", "burned_obscured"], inplace=True)
    res_neg.drop(columns=["shadow", "obscured", "burned_shadow", "burned_obscured"], inplace=True)
    logging.info(f"Shape of positive results: {res_pos.shape}")
    logging.info(f"Shape of negative results: {res_neg.shape}")

    # Calculate stats
    stats_pos = calculate_stats(res_pos)
    stats_neg = calculate_stats(res_neg)
    summary_pos = sum_res(stats_pos)
    summary_neg = sum_res(stats_neg)
    stats_pos = stats_pos.rename(columns={"filename": "positive", 
                                          "ratio_cloud": "pos_ratio_cloud",
                                          "ratio_burned": "pos_ratio_burned",
                                          "ratio_burned_on_clear": "pos_ratio_burned_on_clear",
                                          "ratio_burned_on_cloud": "pos_ratio_burned_on_cloud"})
    stats_neg = stats_neg.rename(columns={"filename": "negative", 
                                          "ratio_cloud": "neg_ratio_cloud",
                                          "ratio_burned": "neg_ratio_burned",
                                          "ratio_burned_on_clear": "neg_ratio_burned_on_clear",
                                          "ratio_burned_on_cloud": "neg_ratio_burned_on_cloud"})
    summary_pos.to_csv(os.path.join(PREDICTION_ANALYSE_PATH, "summary_pos.csv"), index=False)
    summary_neg.to_csv(os.path.join(PREDICTION_ANALYSE_PATH, "summary_neg.csv"), index=False)
    logging.info("Finished calculate stats for positive and negative groups")

    # Paired results with merging
    index = pd.read_csv(os.path.join(DATA_PATH, "index.csv"))
    merge_tmp = index.merge(stats_pos, on="positive", how="inner")
    stats_merge = merge_tmp.merge(stats_neg, on="negative", how="inner")
    stats_merge.to_csv(os.path.join(PREDICTION_ANALYSE_PATH, "stats_merge.csv"), index=False)
    logging.info("Completed save burned stats on each images")

    # Compare pos/neg with visualization
    fig_all, fig_clear, fig_cloud, fig_scatter = compare_visual(stats_merge)
    fig_all.savefig(os.path.join(PREDICTION_ANALYSE_PATH, "boxplot_all.png"))
    fig_clear.savefig(os.path.join(PREDICTION_ANALYSE_PATH, "boxplot_clear.png"))
    fig_cloud.savefig(os.path.join(PREDICTION_ANALYSE_PATH, "boxplot_cloud.png"))
    fig_scatter.savefig(os.path.join(PREDICTION_ANALYSE_PATH, "scatter_corr.png"))
    logging.info("Complete saving visualization for comparison")

    # Hypothesis Test
    stats_merge["diff_ratio_burned"] = stats_merge["pos_ratio_burned"] - stats_merge["neg_ratio_burned"]
    stats_merge["diff_ratio_burned_on_clear"] = stats_merge["pos_ratio_burned_on_clear"] - stats_merge["neg_ratio_burned_on_clear"]

    # on all region
    fig_diff_hist = plot_diff_hist(data=stats_merge["diff_ratio_burned"],
                                   title="Distribution of Differences (all region)")
    fig_diff_hist.savefig(os.path.join(PREDICTION_ANALYSE_PATH, "hist_diff_all.png"))
    sign_npos, sign_nneg, sign_pvalue = sign_test(diff_data=stats_merge["diff_ratio_burned"])
    permute_res = permutation_test(pos_data=stats_merge["pos_ratio_burned"], 
                                   neg_data=stats_merge["neg_ratio_burned"])
    bst_res = bootstrap(diff_data=stats_merge["diff_ratio_burned"])

    # on clear region
    fig_diff_hist_clear = plot_diff_hist(data=stats_merge["diff_ratio_burned_on_clear"],
                                         title="Distribution of Differences (clear region)")
    fig_diff_hist_clear.savefig(os.path.join(PREDICTION_ANALYSE_PATH, "hist_diff_clear.png"))
    sign_npos_clear, sign_nneg_clear, sign_pvalue_clear = sign_test(stats_merge["diff_ratio_burned_on_clear"])
    permute_res_clear = permutation_test(pos_data=stats_merge["pos_ratio_burned_on_clear"], 
                                         neg_data=stats_merge["neg_ratio_burned_on_clear"])
    bst_res_clear = bootstrap(diff_data=stats_merge["diff_ratio_burned_on_clear"])

    # report into txt file
    report = f"""
    Hypothesis Test of Burn Ratio Between Positive and Negative Groups
    ------------------------------------------------------------------
    Number of paired samples: {len(stats_merge)}
    ------------------------------------------------------------------
    On All Region
    Sign Test:
        - n_pos: {sign_npos}
        - n_neg: {sign_nneg}
        - p-value: {sign_pvalue}
    Permutation Test:
        - test statistics (mean): {permute_res.statistic}
        - p-value: {permute_res.pvalue}
    Bootstrap CI:
        - CI_low: {bst_res.confidence_interval.low}
        - CI_high: {bst_res.confidence_interval.high}
        - standard error: {bst_res.standard_error}
    ------------------------------------------------------------------
    On Clear Region
    Sign Test:
        - n_pos: {sign_npos_clear}
        - n_neg: {sign_nneg_clear}
        - p-value: {sign_pvalue_clear}
    Permutation Test:
        - test statistics (mean): {permute_res_clear.statistic}
        - p-value: {permute_res_clear.pvalue}
    Bootstrap CI:
        - CI_low: {bst_res_clear.confidence_interval.low}
        - CI_high: {bst_res_clear.confidence_interval.high}
        - standard error: {bst_res_clear.standard_error}
    """

    with open(os.path.join(PREDICTION_ANALYSE_PATH, "hypothesis_test.txt"), "w") as f:
        f.write(report)
    logging.info("Completed do hypothesis test and save results")