"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

import numpy as np
import pandas as pd
from typing import Tuple 
import os


# Extract top predictions
def TopPred(top_ranks = 7,
            metric = 'mean_spearman_test',
            strat_method = 'expmethod_kmeans',
            fp_database = './stabpred/training/dataset_640/database_640.pkl',
            perf_csv_dir = "./stabpred/training/scripts/performances",
            saved_predictions_path = './stabpred/training/scripts/predictions',
            do_diverse_sorting = True,
            diverse_emb = False,
            diverse_reg = False,
            seed_idx: int=0,
            ) -> Tuple[list, list]:
    
    '''Will select the predictions from multiple models (in saved_predictions_path) 
    based on the performances in perf_csv_dir.

    If multiple seeds were gicen in StratifiedNestCV, the individual models should have generated the same amount of predictions for each input.
    seed_idx will pick the corresponding set of predicted temperatures.

    Selection of the models is done based on do_diverse_sorting, diverse_emb and diverse_reg (see parameters)

    Parameters
    ----------
        - top_ranks: int
            The number of top ranking models to selection predictions from.

        - metric: str
            The scoring metric to rank the models on.
              (e.g., 'mean_spearman_test' or 'mean_pearson_test')
        
        - strat_method: str
            Stratification method used during training (to seek for the correct
            model in the saved performances and predictions).
        
        - fp_database: str
            filepath to the database.
        
        - perf_csv_dir: str
            File path to the saved performances for ranking.

        - saved_predictions_path: str
            Filepath to the saved predictions. 

        - do_diverse_sorting: bool
            If True, will take the best model for each embedding (if diverse_emb),
            or for each regression (if diverse_reg).
            If False, will take the absolute ranking independently on the type of 
            embedding and regression. 

        - diverse_emb: bool
            If True, will take the best model for each embedding. 

        - diverse_reg: bool
            If True, will take the best model for each regression. 

        - seed_idx: int
            The seed idx we are considering [0, +inf[. If 1, will take the second saved
            predicted temperature from the individual model. If tehre is 
            only one, just put zero

    Returns
    ----------
        - predictions: list of list 
            List of predictions (top_ranks) for each datapoint in database. 
    '''

    rank_by = f'mean_{metric}_test'
    database = pd.read_pickle(fp_database)

    # Compile all the performances from the different models (different files)
    regressors, embeddings, p_metric_values, p_stds, s_metric_values, s_stds, r_metric_values, r_stds = [], [], [], [], [], [], [], []
    sdr_metric_values, sdr_stds = list(), list()
    strat = []
    for filename in os.listdir(perf_csv_dir):
        if filename.endswith("_perf.csv"):
            csv_path = os.path.join(perf_csv_dir, filename)
            perf_df = pd.read_csv(csv_path)
            strat.extend(perf_df['strat_method'].values)
            
            regressors.extend([filename.replace('_perf.csv','')]*len(perf_df.index))
            embeddings.extend(perf_df['embeddings'].values)
            p_metric_values.extend(perf_df['mean_pearson_test'].values)
            p_stds.extend(perf_df['std_pearson_test'].values)
            s_metric_values.extend(perf_df['mean_spearman_test'].values)
            s_stds.extend(perf_df['std_spearman_test'].values)
            r_metric_values.extend(perf_df['mean_mae_test'].values)
            r_stds.extend(perf_df['std_mae_test'].values)
            sdr_metric_values.extend(perf_df['mean_sdr_test'].values)
            sdr_stds.extend(perf_df['std_sdr_test'].values)

    df = pd.DataFrame({'reg': regressors,
					'embed': embeddings,
                    'strat_method': strat,
					f'mean_pearson_test': [round(s,3) for s in p_metric_values],
                    f'std_pearson_test': [round(s,3) for s in p_stds],
                    f'mean_spearman_test': [round(s,3) for s in s_metric_values],
                    f'std_spearman_test': [round(s,3) for s in s_stds],
                    f'mean_mae_test': [round(s,2) for s in r_metric_values],
                    f'std_mae_test': [round(s,2) for s in r_stds],
                    f'mean_sdr_test': [round(s,2) for s in sdr_metric_values],
                    f'std_sdr_test': [round(s,2) for s in sdr_stds]
                    })

    df_strat = df[df["strat_method"]==strat_method]



    # Sort by test performance
    if rank_by == "mean_mae_test":
        ascending = True
    else:
        ascending = False
    df_sorted = df_strat.sort_values(by=rank_by, ascending=ascending).reset_index().drop(columns=['index'])

    # Filter by diverse embeddings or regressors
    if do_diverse_sorting:
        if diverse_emb:
            df_sorted = df_sorted.drop_duplicates(subset='embed', keep='first')
        else:
            if diverse_reg:
                df_sorted = df_sorted.drop_duplicates(subset='reg', keep='first')
            else:
                df_sorted = df_sorted

    
    print(f"Sort by {rank_by}")
    print(df_sorted.shape)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_sorted[:top_ranks])

    # Save the predictions by the [regressor, embed] combinations in top_ranks
    predictions = [[] for _ in range(len(database))]

    for i in range(int(top_ranks)):
        reg = df_sorted['reg'].values[i]
        embed = df_sorted['embed'].values[i]
        
        # All data
        csv_path = os.path.join(saved_predictions_path, f'{reg}_pred.csv')
        perf_df = pd.read_csv(csv_path)
        saved_pred = perf_df[f'{embed}_{strat_method}'].values
        
        for j in range(len(predictions)):
            predictions[j].append(float(saved_pred[j].split('_')[seed_idx]))

    return np.array(predictions)


