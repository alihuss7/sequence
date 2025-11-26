"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""


import pandas as pd
import csv
import os

import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error

from .modules.strat_nest import StratifiedNestedCV
from .modules.ensemble import TopPred

if __name__ == "__main__":

    # Stratnested parameters
    strat_method = "expmeth_clust_kmedoids_blosum" # 'expmeth_clust_kmedoids_blosum' 'exp_method'
    regression = "ridge"
    num_outer_fold = 3
    num_inner_fold = 3
    random_states = [0,
                    42,
                    100
                     ]
    
    # Extract top predictions
    metric = "spearman" # or "pearson" or "mae"
    
    # Choose diverse embedding or regression
    do_diverse_sorting = True
    diverse_emb = True
    diverse_reg = False

    # Set number of features representing tm and exp method in the ensembled embedding
    num_total_temperatures = 8 # number of top temperatures selected for ensemble, need >= 1
    if do_diverse_sorting and diverse_reg: num_total_temperatures = 7

    # File paths
    saved_embed_path = "../../data/database_640.pkl"
    perf_csv_dir = "./saved_results/performances_exp_blosum_640_mae_no_mean"
    saved_predictions_path = "./saved_results/predictions_exp_blosum_640_mae_no_mean"   

    if not do_diverse_sorting: id = 'true_top'
    elif diverse_emb: id = 'div_emb'
    else: id = 'div_reg'

    if not os.path.exists(perf_csv_dir):
        os.makedirs(perf_csv_dir)
    fp_ensemb_metrics = os.path.join(perf_csv_dir, f"ens_no_mean_{metric}_{id}.csv")

    if not os.path.exists(saved_predictions_path):
        os.makedirs(saved_predictions_path)
    fp_predictions = os.path.join(saved_predictions_path, f"ens_no_mean_{metric}_{id}.csv")

    embed_df = pd.read_pickle(saved_embed_path)

    if os.path.exists(fp_ensemb_metrics): os.remove(fp_ensemb_metrics)
    if os.path.exists(fp_predictions): os.remove(fp_predictions)

    for nb_temp in range(1,num_total_temperatures+1):

        trial = f"tm_{nb_temp}"

        # Loop over all embeddings
        list_fin_pearson_train = []
        list_fin_spearman_train = []
        list_fin_RSME_train = []
        list_fin_SDR_train = []

        list_fin_pearson_test = []
        list_fin_spearman_test = []
        list_fin_RSME_test = []
        list_fin_SDR_test = []

        list_pred = []

        for seed_idx, seed in enumerate(random_states): 

            # Extract set of predicitons for ensembling
            data = TopPred(top_ranks=nb_temp, 
                                    strat_method=strat_method,
                                    metric=metric,
                                    fp_database=saved_embed_path,
                                    saved_predictions_path=saved_predictions_path,
                                    perf_csv_dir=perf_csv_dir,
                                    diverse_emb=diverse_emb,
                                    diverse_reg=diverse_reg,
                                    do_diverse_sorting=do_diverse_sorting,
                                    seed_idx=seed_idx)
            
            target = embed_df['tm'].values
            label = embed_df[f'{strat_method}'].values
            print("Full dataset: ", data.shape, target.shape, label.shape)

            # Save evaluation metrics
            train_metrics, test_metrics, list_predictions = StratifiedNestedCV(
                            data, target, label,
                            num_outer_fold, num_inner_fold, [seed], regression,
                            strat_method = strat_method,
                            return_averaged=False
                            )

            # Save perfs by seed
            list_fin_pearson_train.append(train_metrics[0])
            list_fin_spearman_train.append(train_metrics[3])
            list_fin_RSME_train.append(train_metrics[6])
            list_fin_SDR_train.append(train_metrics[8])

            list_fin_pearson_test.append(test_metrics[0])
            list_fin_spearman_test.append(test_metrics[3])
            list_fin_RSME_test.append(test_metrics[6])
            list_fin_SDR_test.append(test_metrics[8])

            list_pred.append(list_predictions[0])

            # GET BASELINE
            datatop = TopPred(top_ranks=1, 
                                strat_method=strat_method,
                                metric=metric,
                                fp_database=saved_embed_path,
                                saved_predictions_path=saved_predictions_path,
                                perf_csv_dir=perf_csv_dir,
                                diverse_emb=diverse_emb,
                                diverse_reg=diverse_reg,
                                do_diverse_sorting=do_diverse_sorting,
                                seed_idx=seed_idx)

            # Get performance top model
            pearson, pearson_p = scipy.stats.pearsonr(datatop[:,0], target)
            spearman, spearman_p = scipy.stats.spearmanr(datatop[:,0], target)
            mae = np.sqrt(mean_squared_error(datatop[:,0], target))
            print('[BASELINE] TOP PERFORMING PERF ON WHOLE')
            print('pearson: ', pearson, ' spearman: ', spearman, ' mae: ',mae)

        pearson_mean_train, pearson_std_train = np.mean(list_fin_pearson_train), np.std(list_fin_pearson_train)
        spearman_mean_train, spearman_std_train = np.mean(list_fin_spearman_train), np.std(list_fin_spearman_train)
        RSME_mean_train, RSME_std_train = np.mean(list_fin_RSME_train), np.std(list_fin_RSME_train)
        SDR_mean_train, SDR_std_train = np.mean(list_fin_SDR_train), np.std(list_fin_SDR_train)

        train_metrics = [pearson_mean_train, pearson_std_train, spearman_mean_train, spearman_std_train,
                         RSME_mean_train, RSME_std_train, SDR_mean_train, SDR_std_train]

        pearson_mean_test, pearson_std_test = np.mean(list_fin_pearson_test), np.std(list_fin_pearson_test)
        spearman_mean_test, spearman_std_test = np.mean(list_fin_spearman_test), np.std(list_fin_spearman_test)
        RSME_mean_test, RSME_std_test = np.mean(list_fin_RSME_test), np.std(list_fin_RSME_test)
        SDR_mean_test, SDR_std_test = np.mean(list_fin_SDR_test), np.std(list_fin_SDR_test)

        test_metrics = [pearson_mean_test, pearson_std_test, spearman_mean_test, spearman_std_test,
                         RSME_mean_test, RSME_std_test, SDR_mean_test, SDR_std_test]
  
        # Save metrics and predictions on whole dataset
        header = ['method','trial',
                'fin_pearson_train', 'std_pearson_train', 
                'fin_spearman_train', 'std_spearman_train',
                'fin_mae_train', 'std_mae_train', 
                'fin_std_ratio_train', 'std_std_ratio_train',
                'fin_pearson_test','std_pearson_test',
                'fin_spearman_test', 'std_spearman_test',
                'fin_mae_test','std_mae_test',  
                'fin_std_ratio_test', 'std_std_ratio_test',]

        metrics = []
        metrics.extend([regression, trial])
        metrics.extend(train_metrics)
        metrics.extend(test_metrics)
        
        if not os.path.exists(fp_ensemb_metrics):    
            with open(fp_ensemb_metrics, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header) 
                csv_writer.writerow(metrics)
        else:
            with open(fp_ensemb_metrics, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(metrics)
                
        if not os.path.exists(fp_predictions):
            predictions = pd.DataFrame()
            predictions[f'{regression}_{trial}'] = ['_'.join([str(tm) for tm in preds]) for preds in np.array(list_pred).T]
            predictions.to_csv(fp_predictions,index=False,header=True)
        else:
            predictions = pd.read_csv(fp_predictions)
            predictions[f'{regression}_{trial}'] = ['_'.join([str(tm) for tm in preds]) for preds in np.array(list_pred).T]
            predictions.to_csv(fp_predictions,index=False,header=True)

            


