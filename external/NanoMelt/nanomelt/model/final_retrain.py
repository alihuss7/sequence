"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import joblib
import os

from .modules.strat_nest import RegressionFit
from .modules.utils import perf_metrix
from .modules.strat_nest import CustomLeaveOneClassOut, CustomStratifiedKFold


# Hyperparameter tuning
def hp_tuning(database, data_train, target_train, label_train, reg, 
              num_folds, strat_method, folder_save, fn_pred, model_name,
              random_states = [0], feature_sel_method='SelectFromModel',
              name_col_target = 'tm'):
    """
    Hyperparameter tuning for the regression models.
    If multiple random_states, will run Cv multiple times
    for each model for hyper-parameter tuning. 
    Yet will take an average of the predictions temperatures, which
    is not ideal. For NanoMelt, the model was finally retrained on the whole
    dataset with one seed only. 

    Parameters
    ----------
    database : pd.DataFrame
        The database containing the embeddings and target values.
    data_train : np.array
        The training data.
    target_train : np.array
        The target values.
    label_train : np.array
        The stratification labels.
    reg : str
        The regression model.
    num_folds : int
        The number of folds used in the hyperparameter tuning.
    strat_method : str
        The stratification method used.
    folder_save : str
        The folder where to save ebverything. 
    fn_pref:
        The filename to saved predictions.
    model_name : str
        The name of the model to be saved.
    random_states : list
        The random states used in the hyperparameter tuning.
    feature_sel_method : str
        The feature selection method used.
    """
    

    list_score_dict, list_pred = [], []

    # Find the best model for each seed
    for seed in tqdm(random_states):
        
        inner_loop = RegressionFit(data_train, target_train, label_train,
                    feature_sel_method = feature_sel_method, GridSearch_folds = num_folds, 
                    random_state = seed, regression = reg, strat_method = strat_method)

        param_scores, pipe = inner_loop.param_scores()

        list_score_dict.append(param_scores)
        pred = [None]*len(data_train)

        if strat_method == 'none':
            splits = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        elif strat_method == 'loco':
            splits = CustomLeaveOneClassOut()
        else:
            splits = CustomStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

        # Predict on the test set for each fold
        for fold, (train_index, test_index) in enumerate(splits.split(data_train, target_train, groups=label_train)):

            # Rank the train scores for each fold
            inner_train_score = param_scores[f"split{fold}_test_score"]
            sorted_train = np.argsort(inner_train_score)[::-1]
            rank_list = np.empty_like(sorted_train)
            rank_list[sorted_train] = np.arange(1, len(inner_train_score) + 1)

            # Find the best params for each fold
            param_scores[f"split{fold}_rank_train_score"] = rank_list
            param_scores = pd.DataFrame(param_scores)
            best_params = param_scores.loc[param_scores[f"split{fold}_rank_train_score"]==1, "params"].values[0]
            pipe.set_params(**best_params)

            # Fit the model and predict on the test set
            pipe.fit([data_train[i] for i in train_index], [target_train[i] for i in train_index])
            pred_fold = pipe.predict([data_train[i] for i in test_index])
            
            # Store the predictions for each test set
            for i in range(len(test_index)):
                test_id = test_index[i]
                pred_tm = pred_fold[i]
                pred[test_id] = pred_tm

        list_pred.append(pred) # Store the test predictions for each seed

    predictions = np.mean(np.array(list_pred).T, axis=1) # Average the predictions across all seeds

    # Print the evaluation metrics
    pearson_train, _, spearman_train, _, mae_train, std_ratio_train = perf_metrix(predictions,target_train)
    print('Tuned overall test perf', pearson_train, spearman_train, mae_train, std_ratio_train)

    # Save the [TEST] predictions
    fp_pred = os.path.join(folder_save, fn_pred)

    if not os.path.exists(fp_pred):
        pred_df = pd.DataFrame({'id': database['id'].to_list(),name_col_target: database[name_col_target].to_list()})
        pred_df[f"{model_name}"] = predictions
        pred_df.to_csv(fp_pred, index=False)
    else:
        pred_df = pd.read_csv(fp_pred)
        pred_df[f"{model_name}"] = predictions
        pred_df.to_csv(fp_pred, index=False)

    # Average the mean_test_score for each param across all seeds
    list_mean_test_score = list()
    for score_dict in list_score_dict:
        list_mean_test_score.append(np.array(score_dict['mean_test_score']))
    inner_mean_test_score = [np.mean(k) for k in zip(*list_mean_test_score)]

    # Rank params according to mean_test_score with the highest score ranked as 1
    sorted_indices = np.argsort(inner_mean_test_score)[::-1]
    rank_list = np.empty_like(sorted_indices)
    rank_list[sorted_indices] = np.arange(1, len(inner_mean_test_score) + 1)

    # Store mean_test_score and rank_test_score for each param across all seeds
    param_dict = {}
    param_dict['params'] = list_score_dict[0]['params']
    param_dict['mean_test_score'] = inner_mean_test_score
    param_dict['rank_test_score'] = rank_list

    # Find best params across all seeds
    param_df = pd.DataFrame(param_dict)
    best_params = param_df.loc[param_df['rank_test_score']==1, 'params'].values[0]
    pipe.set_params(**best_params)
    print(f"\nTuned params for {model_name}: \n{pipe}")

    # Train and predict with the best model
    pipe.fit(data_train, target_train)
    pred_train = pipe.predict(data_train)
    joblib.dump(pipe, os.path.join(folder_save, f'{model_name}.pkl'))

    

# Train the submodels
def train_submodels(database, list_submodels, strat_method, 
                    folder_save, num_folds=3, name_col_target = 'tm'):
    """
    Train the individual submodels for each regression model and embedding.

    Parameters
    ----------
    database : pd.DataFrame
        The database containing the embeddings and target values.
    list_submodels : list
        List containing tuples (in order) (reg, emb).
    strat_method : str
        Name of the classs label to do the stratification on.
    folder_save : str
        The file path to the saved predictions.
    num_folds : int
        Nb of folds for repeat
    """

    fn_pred = 'submodels_pred.csv'

    # Define the target and labels
    target_train = database[name_col_target]
    label_train = database[f'{strat_method}']

    # Train and predict with the best model
    for reg, emb in tqdm(list_submodels):
        data_train = np.vstack(database[f"{emb}"])
        hp_tuning(database, data_train, target_train, label_train, reg, num_folds, strat_method, folder_save, fn_pred, model_name=f"{reg}_{emb}", name_col_target=name_col_target)



def train_ensemb(database, list_submodels, strat_method, folder_save, num_folds=3, name_col_target = 'tm'):
     
    """
    Train the ensemble model using the best predictions from the submodels.
    
    Parameters
    ----------
    database : pd.DataFrame
        The database containing the embeddings and target values.
    list_submodels : list
        List containing tuples (in order) (reg, emb).
    fp_pred : str
        The file path to the saved predictions.
    strat_method : str
        Name of the classs label to do the stratification on.
    folder_save : str
        The file path to the saved predictions.
    num_folds : int
        Nb of folds for repeat
    """

    # Load the saved predictions
    fp_pred_submodels = os.path.join(folder_save, 'submodels_pred.csv')
    saved_pred = pd.read_csv(fp_pred_submodels)

    cols = list()
    for reg, emb in list_submodels:
        cols.append(f'{reg}_{emb}')

    saved_pred = saved_pred[cols]
    print(saved_pred.columns)

    # Define the training data
    data_train = saved_pred.values
    target_train = database[name_col_target]
    label_train = database[f'{strat_method}']

    fn_pred = 'ensemb_pred.csv'

    # Train model
    reg = 'ridge'
    hp_tuning(database, data_train, target_train, label_train, reg, num_folds, strat_method, folder_save, fn_pred, model_name=f"{reg}_ensemb", name_col_target=name_col_target)



def PipeTrain(df_embed: pd.DataFrame, 
              strat_method:str = 'expmeth_clust_kmedoids_blosum',
              list_submodels:list = [('SVR', 'esm1b'), ('SVR', 'esm2_t30')],
              folder_save:str = 'saved_folder', 
              name_col_target = 'tm',
              num_folds:int = 3):
    """
    Train individual regression models and ensemble them into a final model. 
    This pipeline is to train a deployable model (no repeated evaluations, just one seed).
    Stratifie cross-validation is applied to tune the models to find the best
    hyperparameters before a global refit.

    Parameters
    ----------
    - df_embed: pd.DataFrame
            Filepath to the fasta file of input sequences to compute the embeddings on.
    - strat_method : str
        Name of the classs label to do the stratification on
    - list_submodels : list
        List containing tuples (in order) (reg, emb).
    - folder_save : str
        The filepath of the folder where to save the saved models. If doesn't exist will create one
    - num_folds : int
        Number of CV folds
    """

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    # Train the submodels
    train_submodels(df_embed, list_submodels, strat_method, folder_save, num_folds, name_col_target)

    # Train the ensemble model
    train_ensemb(df_embed, list_submodels, strat_method, folder_save, num_folds, name_col_target)

    return



if __name__ == '__main__':
    # Arguments for final published NanoMelt paper 
    fp_database ='../../data/database_640.pkl'
    strat_method = 'expmeth_clust_kmedoids_blosum'
    folder_save = 'saved_models/NanoMelt_finalmodel'
    list_submodels = [('SVR', 'esm1b'),
                ('SVR', 'esm2_t30'),
                ('GPR', 'vhse'),
                ('GPR', 'onehot'),
                ]
    
    df_embed = pd.read_pickle(fp_database)

    # Final retrain
    PipeTrain(df_embed, strat_method=strat_method, 
              list_submodels=list_submodels,folder_save=folder_save)