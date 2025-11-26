"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

import pandas as pd
import numpy as np
import os

from joblib.parallel import Parallel, delayed

from .modules.strat_nest import StratifiedNestedCV



def reg_parall_as_job(reg, saved_embed_path, fp_predictions_save, fp_reg_metrics_save, 
                      embeddings, sel_method, strat_method, 
                      random_state, num_outer_fold, num_inner_fold):
    
    if not os.path.exists(fp_predictions_save):
        os.makedirs(fp_predictions_save)
    fp_predictions_df = os.path.join(fp_predictions_save, f"{reg}_pred.csv")

    if not os.path.exists(fp_reg_metrics_save):
        os.makedirs(fp_reg_metrics_save)
    fp_reg_metrics = os.path.join(fp_reg_metrics_save, f"{reg}_perf.csv")

    embed_df = pd.read_pickle(saved_embed_path)
    print(embed_df.columns)

    if not os.path.exists(fp_predictions_df):
        df = embed_df[['id','tm']]
        df.to_csv(fp_predictions_df,index=False,header=True)

    # Data Target Loading
    target = embed_df['tm'].values
    if strat_method == 'none':
        label = embed_df['tm'].values
    elif strat_method == 'loco':
        label = embed_df['clust_kmedoids_blosum'].values
    else:
        label = embed_df[f'{strat_method}'].values

    # Loop over all embeddings
    list_class,list_embed,list_sel_method = [],[],[]

    list_fin_pearson_train, list_fin_pearson_p_train, list_std_pearson_train = [],[],[]
    list_fin_spearman_train, list_fin_spearman_p_train, list_std_spearman_train = [],[],[]
    list_fin_RSME_train, list_std_RSME_train = [],[]
    list_fin_SDR_train, list_std_SDR_train = [],[]

    list_fin_pearson_test, list_fin_pearson_p_test, list_std_pearson_test = [],[],[]
    list_fin_spearman_test, list_fin_spearman_p_test, list_std_spearman_test = [],[],[]
    list_fin_RSME_test, list_std_RSME_test = [],[]
    list_fin_SDR_test, list_std_SDR_test = [],[]

    list_num_seed = []

    for embed_method in embeddings:
        print(f"\n-->> ENCODING sequences with {embed_method} <<--\n")
        data = embed_df[f'{embed_method}'].values

        train_metrics, test_metrics, list_predictions = StratifiedNestedCV(
                                data, target, label,
                                num_outer_fold, num_inner_fold, 
                                random_state, regression=reg, 
                                feature_sel_method=sel_method, 
                                strat_method=strat_method)

        list_class.append(strat_method)
        list_embed.append(embed_method)
        list_sel_method.append(sel_method)

        list_fin_pearson_train.append(train_metrics[0])
        list_fin_pearson_p_train.append(train_metrics[1])
        list_std_pearson_train.append(train_metrics[2])
        list_fin_spearman_train.append(train_metrics[3])
        list_fin_spearman_p_train.append(train_metrics[4])
        list_std_spearman_train.append(train_metrics[5])
        list_fin_RSME_train.append(train_metrics[6])
        list_std_RSME_train.append(train_metrics[7])
        list_fin_SDR_train.append(train_metrics[8])
        list_std_SDR_train.append(train_metrics[9])

        list_fin_pearson_test.append(test_metrics[0])
        list_fin_pearson_p_test.append(test_metrics[1])
        list_std_pearson_test.append(test_metrics[2])
        list_fin_spearman_test.append(test_metrics[3])
        list_fin_spearman_p_test.append(test_metrics[4])
        list_std_spearman_test.append(test_metrics[5])
        list_fin_RSME_test.append(test_metrics[6])
        list_std_RSME_test.append(test_metrics[7])
        list_fin_SDR_test.append(test_metrics[8])
        list_std_SDR_test.append(test_metrics[9])

        list_num_seed.append(len(random_state))

        # Save predictions
        predictions_df = pd.read_csv(fp_predictions_df)
        col = f"{embed_method}_{strat_method}"
        predictions_df[col] = ['_'.join([str(tm) for tm in preds]) for preds in np.array(list_predictions).T]
        predictions_df.to_csv(fp_predictions_df, index=False, header=True)

    # Dictionary of evaluation metrics
    performance_dict = {'strat_method': list_class,
                        'num_seed': list_num_seed, 
                        'embeddings': list_embed,
                        'feature_sel': list_sel_method,
                        
                        'mean_pearson_train': list_fin_pearson_train,
                        'mean_pearson_p_train': list_fin_pearson_p_train,
                        'std_pearson_train': list_std_pearson_train,
                        'mean_pearson_test': list_fin_pearson_test,
                        'mean_pearson_p_test': list_fin_pearson_p_test,
                        'std_pearson_test': list_std_pearson_test,
                                                
                        'mean_spearman_train': list_fin_spearman_train,
                        'mean_spearman_p_train': list_fin_spearman_p_train,
                        'std_spearman_train': list_std_spearman_train,
                        'mean_spearman_test': list_fin_spearman_test,
                        'mean_spearman_p_test': list_fin_spearman_p_test,
                        'std_spearman_test': list_std_spearman_test,
                                                
                        'mean_mae_train': list_fin_RSME_train,
                        'std_mae_train': list_std_RSME_train,
                        'mean_mae_test': list_fin_RSME_test,
                        'std_mae_test': list_std_RSME_test,
                        
                        'mean_sdr_train': list_fin_SDR_train,
                        'std_sdr_train': list_std_SDR_train,
                        'mean_sdr_test': list_fin_SDR_test,
                        'std_sdr_test': list_std_SDR_test
                        }
    
    performance_df = pd.DataFrame(performance_dict)
    print(f"\nPerformance metrics: {performance_df}")

    # Save metrics to csv
    if not os.path.exists(fp_reg_metrics):
        performance_df.to_csv(fp_reg_metrics, mode='w', index=False, header=True)
    else:
        performance_df.to_csv(fp_reg_metrics, mode='a', index=False, header=False)



if __name__ == "__main__":
    
    # File paths
    saved_embed_path = '../../data/database_640.pkl'

    # Folder where to save performances and predictions
    fp_predictions_save = "./saved_results/predictions_exp_blosum_640_mae_no_mean" 
    fp_reg_metrics_save = './saved_results/performances_exp_blosum_640_mae_no_mean'

	# Parameters for trial
    num_outer_fold = 3
    num_inner_fold = 3
    random_state = [0, 42, 100]

    regressions = [
                    'ridge',
        			 'huber',
                	 'GPR',
                	 'SVR',
                	 'elasticnet',
                	 'RF',
                     'LightGBM',
                   ]
    
    embeddings = [
                   'onehot',
                   'esm1b',
                   'esm2_t30',
                   'antiberty',
                   'ablang',
                    'nanobuilder',
                    'vhse',
                 'nanobert',
                  ]
    
    feature_sel_method = 'SelectFromModel' # ['SelectFromModel','PCA', None]
    
    strat_methods = ['expmeth_clust_kmedoids_blosum']

    # Evaluate single regression models, loop over all regressions
    for strat_method in strat_methods:
        Parallel(n_jobs=-1)(delayed(reg_parall_as_job)(reg, saved_embed_path,
                                                    fp_predictions_save, fp_reg_metrics_save,
                                                    embeddings, feature_sel_method, 
                                                    strat_method, random_state,
                                                    num_outer_fold, num_inner_fold) 
                                            for reg in regressions)
    