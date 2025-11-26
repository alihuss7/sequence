"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

import numpy as np
import pandas as pd
import os

from .modules.ensemble import TopPred
from .modules.utils import perf_metrix

if __name__ == "__main__":

    # File paths
    saved_embed_path = "../../data/database_640.pkl"
    strat_method = 'expmeth_clust_kmedoids_blosum' # 'expmeth_clust_kmedoids_blosum' 'exp_method'

    perf_csv_dir = "./saved_results/performances_exp_blosum_640_mae_no_mean"
    saved_predictions_path = "./saved_results/predictions_exp_blosum_640_mae_no_mean"

    metric = "spearman"

    # Choose diverse embedding or regression
    
    do_diverse_sorting = False
    diverse_emb = False
    diverse_reg = True

    if not do_diverse_sorting: id = 'true_top'
    elif diverse_emb: id = 'div_emb'
    else: id = 'div_reg'

    if not os.path.exists(perf_csv_dir):
        os.makedirs(perf_csv_dir)
    mean_metrics_csv = os.path.join(perf_csv_dir, f"ens_no_mean_{metric}_ave_{id}.csv")

    if not os.path.exists(saved_predictions_path):
        os.makedirs(saved_predictions_path)
    mean_pred_csv = os.path.join(saved_predictions_path, f"ens_no_mean_{metric}_ave_{id}.csv")

    if os.path.exists(mean_metrics_csv): os.remove(mean_metrics_csv)
    if os.path.exists(mean_pred_csv): os.remove(mean_pred_csv)

    embed_df = pd.read_pickle(saved_embed_path)



    # Number of top ranks to average
    num_to_ave = 8

    # parameters
    
    num_tm_list = np.arange(1,num_to_ave+1,1)

    seeds = [0,42,100]

    for num_tm in num_tm_list:
        saved_pearson, saved_spearman, saved_mae, saved_sdr = list(), list(), list(), list()
        list_preds_per_seed = list()
        for seed_idx in range(len(seeds)):
            trial = f"ave_top_{num_tm}"

            predictions = TopPred(top_ranks=num_tm,
                                    metric=metric,
                                    strat_method=strat_method,
                                    fp_database = saved_embed_path, 
                                    perf_csv_dir = perf_csv_dir,
                                    saved_predictions_path = saved_predictions_path, 
                                    diverse_emb=diverse_emb,
                                    diverse_reg=diverse_reg,
                                    do_diverse_sorting=do_diverse_sorting,
                                    seed_idx=seed_idx
                                    )

            ave_pred = [np.mean(i) for i in predictions]
            list_preds_per_seed.append(ave_pred)

            true_tm = embed_df['tm'].values
            id = embed_df['id'].values

            pearson, _, spearman, _, mae, sdr  = perf_metrix(ave_pred,true_tm)
            saved_pearson.append(pearson)
            saved_spearman.append(spearman)
            saved_mae.append(mae)
            saved_sdr.append(sdr)

        mean_pearson, std_pearson = np.mean(saved_pearson), np.std(saved_pearson)
        mean_spearman, std_spearman = np.mean(saved_spearman), np.std(saved_spearman)
        mean_mae, std_mae = np.mean(mae), np.std(mae)
        mean_sdr, std_sdr = np.mean(saved_sdr), np.std(saved_sdr)

        # Save results to csv
        performance_df = {'method': [trial], 'pearson': [mean_pearson], 'std_pearson': [std_pearson], 
                          'spearman': [mean_spearman], 'std_spearman': [std_spearman], 
                          'mae': [mean_mae], 'std_mae': [std_mae],
                          'sdr': [mean_sdr], 'std_sdr': [std_sdr]}
        
        performance_df = pd.DataFrame(performance_df)
        if not os.path.exists(mean_metrics_csv):
            performance_df.to_csv(mean_metrics_csv, mode='w', index=False, header=True)
        else:
            performance_df.to_csv(mean_metrics_csv, mode='a', index=False, header=False)
        
        if not os.path.exists(mean_pred_csv):
            pred_df = {'id': id, 'tm': true_tm, f'{trial}': ['_'.join([str(tm) for tm in preds]) for preds in np.array(list_preds_per_seed).T]}
            pred_df = pd.DataFrame(pred_df)
            pred_df.to_csv(mean_pred_csv, index=False)
        else:
            pred_df = pd.read_csv(mean_pred_csv)
            pred_df[f'{trial}'] = ['_'.join([str(tm) for tm in preds]) for preds in np.array(list_preds_per_seed).T]
            pred_df.to_csv(mean_pred_csv, index=False)
