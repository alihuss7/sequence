"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from typing import Tuple

from .regs import regression_dict
from .utils import perf_metrix


class CustomStratifiedKFold(BaseCrossValidator):
    """
    Stratified K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class. It's a variation of StratifiedKFold that
    takes into account groups as group label and not y. 

    Parameters
    ----------
    - n_splits : int, default=5

    - shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.

    - random_state : int, default=0
        Pass an int for reproducible output across multiple function calls.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, group=None):
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Parameters
        --------
        - X: array-like, shape (n_samples, n_features), optional
            Input data, not used in this implementation.
        - y: array-like, shape (n_samples,), optional
            Target variable, not used in this implementation.
        - group: array-like, shape (n_samples,), optional
            Separate set of labels for stratification.

        Returns
        --------
        - n_splits: int
            The number of splitting iterations in the cross-validator.
        """
        return self.n_splits
    
    def _make_test_folds(self, X, y, groups=None):
        # Convert lists to NumPy arrays if they are not already
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups) if groups is not None else None

        # Ensure that X, y, and group have the same length
        if len(X) != len(y):
            raise ValueError("Input data and target variable must have the same length.")
        if groups is not None and len(groups) != len(X):
            raise ValueError("Group labels must have the same length as input data and target variable.")

        # Use group labels for stratification if provided, otherwise use target variable y
        labels = groups if groups is not None else y

        _, y_idx, y_inv = np.unique(labels, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            print(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # Get class distributions
        uniques, counts = np.unique(labels, return_counts=True)
        props_ori = {unique: round(count/len(labels)*100,2) for unique, count in zip(uniques, counts)}
        #print('\n -> ORIGINAL DATA DISTRIBUTION (in %):', props_ori, '\n')

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(labels), dtype="i")
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                np.random.seed(self.random_state)
                np.random.shuffle(folds_for_class)

            test_folds[y_encoded == k] = folds_for_class

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            test_fold_i = test_folds == i

            # Get class distributions
            if groups is not None:
                test_labels = groups[test_fold_i]
                uniques, counts = np.unique(test_labels, return_counts=True)
                new_props = {unique: round(count/len(test_labels)*100,2) for unique, count in zip(uniques, counts)}
                #print(f'\n -> TEST FOLD-{i} DATA DISTRIBUTION (in %):', new_props, '\n')

            yield test_fold_i


    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """

        return super().split(X, y, groups)


class CustomLeaveOneClassOut(BaseCrossValidator):
    """
    Leave-One-Out cross-validator for each class.

    Provides train/test indices to split data in train/test sets, ensuring each class
    is left out in turn.

    """

    def __init__(self):
        return

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.
        
        Parameters
        ----------
        - X: array-like, shape (n_samples, n_features), optional
            Input data, not used in this implementation.
        - y: array-like, shape (n_samples,), optional
            Target variable for stratification.
        - groups: array-like, shape (n_samples,), optional
            Group labels for stratification.
        
        Returns
        -------
        - n_splits: int
            The number of splitting iterations in the cross-validator.
        """
        return len(np.unique(groups))

    def split(self, X, y, groups):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        - X: array-like, shape (n_samples, n_features)
            The input data.
        - y: array-like, shape (n_samples,)
            The target variable.
        - groups: array-like, shape (n_samples,)
            Group labels for stratification.
        
        Returns
        -------
        - train: ndarray
            The training set indices for that split.
        - test: ndarray
            The testing set indices for that split.
        """
        unique_groups = np.unique(groups)
        for group in unique_groups:
            test_indices = np.where(groups == group)[0]
            train_indices = np.where(groups != group)[0]
            yield train_indices, test_indices


class RegressionFit():

    '''
    For given train and test data and target, fit to the given regression.
    Hyperparameters searched by GridSearchCV with StratifiedKFold using customised labels (e.g. Tm methods).

    Attributes
    --------
    - data_train, target_train, label_train: data and target should be np.ndarray;

    - data_test, target_test, label_test: data and target should be np.ndarray;

    - feature_sel_method: str, default = None
        Any from ['SelectFromModel', 'RFECV', 'PCA', None];

    - GridSearch_folds: int, default = 5
        num of folds performed by GridSearchCV with the train_set;

    - random_state: int, default = 5
        Randome state used for StratifiedKFold. 
        Pass an int for reproducible output across multiple function calls;
    
    - regression: str, default = 'ridge'
        Regression model to fit to the dataset.
    '''

    def __init__(self, data_train, target_train, label_train,
                feature_sel_method = None,
                GridSearch_folds: int = 5,
                random_state: int = 5,
                regression: str = 'ridge',
                strat_method : str = 'class_tm_method'):
        
        self.data_train = data_train
        self.target_train = target_train
        self.label_train = label_train

        self.feature_sel_method = feature_sel_method
        self.GridSearch_folds = GridSearch_folds
        self.random_state = random_state
        self.regression = regression
        self.strat_method = strat_method

        if self.feature_sel_method == None:
            self.pipe = Pipeline([('scale', StandardScaler()),],verbose=False)
            
        else:
            if self.feature_sel_method == 'PCA':
                feature_sel = PCA(n_components=0.6)

            # Feature selection with ridge as default
            if self.feature_sel_method == 'SelectFromModel':
                feature_sel = SelectFromModel(Ridge(alpha=200), prefit=False)
            
            if self.feature_sel_method == 'RFECV':
                feature_sel = RFECV(Ridge(alpha=200), step=0.1, cv=3)

            self.pipe = Pipeline([('scale', StandardScaler()),
                            ('reduce_dims', feature_sel),],verbose=False)

    def param_scores(self, ):
        """
        Fit the given regression to the train dataset.
        Hyperparameters searched by GridSearchCV with StratifiedKFold using customised labels (e.g. Tm methods).

        Returns
        --------
        - param_scores: parameter performance dict from grid_search.cv_results_ 
            (see sklearn.model_selection.GridSearchCV documentation)

        - pipe: sklearn.pipeline.Pipeline object

        Example
        --------
        Example pipe: Pipeline(steps=[
            ('scale', StandardScaler()),
            ('reduce_dims', SelectFromModel(estimator=Ridge(alpha=200))), # preprocessing steps
            ['regr', Ridge(alpha=200)] # regression model with best params from GridSearchCV
            ])
        """
        # Define regression model and param_grid
        regr_model = regression_dict[f'{self.regression}'][0]
        param_grid = regression_dict[f'{self.regression}'][1]
        prefix = 'regr__' # update param_grid format for pipe
        param_grid_updated = {prefix + key: value for key, value in param_grid.items()}
        self.pipe.steps.append(['regr',regr_model])

        # HP tuning with GridSearchCV on the train set
        # default scoring method in GridSearch = default scoring method for the regression; 
        # define scoring to be neg MSE for consistency; negative such that max score gives best model
        if self.strat_method == 'none':
            grid_search = GridSearchCV(self.pipe, param_grid=param_grid_updated, cv=self.GridSearch_folds, scoring='neg_mean_squared_error', n_jobs=-1)
        
        elif self.strat_method == 'loco':
            skf = CustomLeaveOneClassOut()
            grid_search = GridSearchCV(self.pipe, param_grid=param_grid_updated, cv=skf, scoring='neg_mean_squared_error', n_jobs=-1)

        else:
            skf = CustomStratifiedKFold(n_splits=self.GridSearch_folds, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(self.pipe, param_grid=param_grid_updated, cv=skf, scoring='neg_mean_squared_error', n_jobs=-1)

        grid_search.fit(self.data_train, self.target_train, groups = self.label_train)

        # Update best model params
        param_scores = grid_search.cv_results_
        pipe = self.pipe.set_params(**grid_search.best_params_)
        
        return param_scores, pipe
    


def StratifiedNestedCV(data: list, target: list, label: list,
                    num_outer_fold: int = 3,
                    num_inner_fold: int = 3,
                    random_states: list = [0,42,100],
                    regression: str = 'ridge',
                    feature_sel_method: str = 'SelectFromModel',
                    strat_method: str = 'expmethod_kmeans',
                    return_averaged: bool = True,
                    ) -> Tuple[list,list,list]:
    """
    Perform stratified nested CV on the dataset provided.
    Labelling of the groups for stratification is provided through the list "label".  
    "strat_method" tells the program which type of stratification algo to do and save the results as.
        e.g., 'none', 'loco' (leave one class out), or other for full stratification(like 'expmethod_kmeans') 

    If multiple seeds are given, the models will generate the same amount of predictions for each input at each seed repeat
    in list_all_seed_predictions

    Parameters
    --------
    - data: list
        List of inputs
    - target: list
        List of target for the regressions
    - label: list
        List of labels for stratificaiton 
    - num_outer_fold: int
    - num_inner_fold: int
    - random_states: list of int
        Random seeds for multiple repeats, only put one if you do not want repeats
    - regression: str
        regression name to be used (e.g., 'linear','lasso','elasticnet'). See regression_dict in regs.py
    - feature_sel_method: str, default = None
        Any from ['SelectFromModel', 'RFECV', 'PCA', None]
    - strat_method: str
        "strat_method" tells the program which type of stratification algo to do and save the results as.
        e.g., 'none', 'loco' (leave one class out), or other for full stratification(like 'expmethod_kmeans') 
    return_averaged: bool
        If True, will average all the performances over the seeds.

    Returns
    --------
    - train_metrics: if return_averaged, performances will have been averaged each outer layer and each seed, 
        otherwise will save every single perf (will need to mean later then, usefull in stacking.py)
    - test_metrics: same but for test
    - list_all_seed_predictions: list of predictions for each seed

    """
    print(f"==================== Stratified Nested CV with {regression} ====================")
    
    list_all_seed_pearson_train,list_all_seed_spearman_train,list_all_seed_mae_train, list_all_seed_std_ratio_train = [],[],[],[]
    list_all_seed_pearson_p_train,list_all_seed_spearman_p_train = [],[]
    
    list_all_seed_pearson_test,list_all_seed_spearman_test,list_all_seed_mae_test, list_all_seed_std_ratio_test = [],[],[],[]
    list_all_seed_pearson_p_test,list_all_seed_spearman_p_test = [],[]
    
    list_all_seed_predictions = []

    for outer_seed in random_states:
        print(f"\n------------ Seed number = {outer_seed} ------------")
        
        # Outer loop
        list_outer_train_pearson,list_outer_train_spearman,list_outer_train_mae, list_outer_train_std_ratio = [],[],[],[]
        list_outer_train_pearson_p,list_outer_train_spearman_p = [],[]

        list_outer_test_pearson,list_outer_test_spearman,list_outer_test_mae, list_outer_test_std_ratio = [],[],[],[]
        list_outer_test_pearson_p,list_outer_test_spearman_p = [],[]

        list_outer_test_pred = [None]*len(data)

        if strat_method == 'none':
            outer_splits = KFold(n_splits=num_outer_fold, shuffle=True, random_state=outer_seed)
        elif strat_method == 'loco':
            outer_splits = CustomLeaveOneClassOut()
        else:
            outer_splits = CustomStratifiedKFold(n_splits=num_outer_fold, shuffle=True, random_state=outer_seed)

        for fold, (train_index, test_index) in enumerate(outer_splits.split(data, target, groups=label)):
            print(f'\n ===== Fold {fold+1} =====\n')

            data_train, data_test = np.vstack([data[i] for i in train_index]), np.vstack([data[i] for i in test_index])
            target_train, target_test = np.array([target[i] for i in train_index]), np.array([target[i] for i in test_index])
            label_train, label_test = np.array([label[i] for i  in train_index]), np.array([label[i] for i in test_index])

            # Inner loop
            list_score_dict = list()
            for inner_seed in random_states:
                inner_loop = RegressionFit(data_train, target_train, label_train,
                            feature_sel_method = feature_sel_method, GridSearch_folds = num_inner_fold, 
                            random_state = inner_seed, regression = regression, strat_method = strat_method)
                param_scores, pipe = inner_loop.param_scores()
                list_score_dict.append(param_scores)
            
            # Store params scores and mean over all seeds
            list_mean_test_score= list()
            for score_dict in list_score_dict:
                list_mean_test_score.append(np.array(score_dict['mean_test_score']))
            inner_mean_test_score = [np.mean(k) for k in zip(*list_mean_test_score)]
            
            # Rank params according to mean_test_score with the highest score ranked as 1
            sorted_indices = np.argsort(inner_mean_test_score)[::-1]
            rank_list = np.empty_like(sorted_indices)
            rank_list[sorted_indices] = np.arange(1, len(inner_mean_test_score) + 1)
            
			# Collect all trialed params in GridSearch into param_dict
            # Same params tried across all seeds, so only one score_dict is used to access the params
            # Also save the corresponding mean_test_score across all seeds and the ranking
            param_dict = {}
            param_dict['params'] = list_score_dict[0]['params']
            param_dict['mean_test_score'] = inner_mean_test_score
            param_dict['rank_test_score'] = rank_list
            
			# Find best params across all seeds
            param_df = pd.DataFrame(param_dict)
            best_params = param_df.loc[param_df['rank_test_score']==1, 'params'].values[0]
            pipe.set_params(**best_params)
            print(f"\nTuned params for outer fold {fold+1} with seed {outer_seed}: \n{pipe}")
            
			# Train with the best model
            pipe.fit(data_train, target_train)
            
			# Save train and test performance 
            pred_train = pipe.predict(data_train) # Outer Train
            pred_test = pipe.predict(data_test) # Outer Test

            train_pearson, train_pearson_p, train_spearman, train_spearman_p, train_mae, train_std_ratio = perf_metrix(pred_train,target_train)
            test_pearson, test_pearson_p, test_spearman, test_spearman_p, test_mae, test_std_ratio = perf_metrix(pred_test,target_test)

            list_outer_train_pearson.append(train_pearson)
            list_outer_train_pearson_p.append(train_pearson_p)
            list_outer_train_spearman.append(train_spearman)
            list_outer_train_spearman_p.append(train_spearman_p)
            list_outer_train_mae.append(train_mae)
            list_outer_train_std_ratio.append(train_std_ratio)

            list_outer_test_pearson.append(test_pearson)
            list_outer_test_pearson_p.append(test_pearson_p)
            list_outer_test_spearman.append(test_spearman)
            list_outer_test_spearman_p.append(test_spearman_p)
            list_outer_test_mae.append(test_mae)
            list_outer_test_std_ratio.append(test_std_ratio)

			# Save predictions
            for i in range(len(test_index)):
                test_id = test_index[i]
                prediction = pred_test[i]
                list_outer_test_pred[test_id] = prediction

        # Mean over all outer folds and save it into the seed list
        seed_fin_pearson_train = list_outer_train_pearson
        seed_fin_pearson_p_train = list_outer_train_pearson_p
        seed_fin_spearman_train = list_outer_train_spearman
        seed_fin_spearman_p_train = list_outer_train_spearman_p
        seed_fin_mae_train = list_outer_train_mae
        seed_fin_std_ratio_train = list_outer_train_std_ratio

        seed_fin_pearson_test = list_outer_test_pearson
        seed_fin_pearson_p_test = list_outer_test_pearson_p
        seed_fin_spearman_test = list_outer_test_spearman
        seed_fin_spearman_p_test = list_outer_test_spearman_p
        seed_fin_mae_test = list_outer_test_mae
        seed_fin_std_ratio_test = list_outer_test_std_ratio
        print(f"\nTest spearman (outer loop) = {np.mean(seed_fin_spearman_test)} for seed {outer_seed}")

        list_all_seed_pearson_train.append(seed_fin_pearson_train)
        list_all_seed_pearson_p_train.append(seed_fin_pearson_p_train)
        list_all_seed_spearman_train.append(seed_fin_spearman_train)
        list_all_seed_spearman_p_train.append(seed_fin_spearman_p_train)
        list_all_seed_mae_train.append(seed_fin_mae_train)
        list_all_seed_std_ratio_train.append(seed_fin_std_ratio_train)

        list_all_seed_pearson_test.append(seed_fin_pearson_test)
        list_all_seed_pearson_p_test.append(seed_fin_pearson_p_test)
        list_all_seed_spearman_test.append(seed_fin_spearman_test)
        list_all_seed_spearman_p_test.append(seed_fin_spearman_p_test)
        list_all_seed_mae_test.append(seed_fin_mae_test)
        list_all_seed_std_ratio_test.append(seed_fin_std_ratio_test)

        list_all_seed_predictions.append(list_outer_test_pred)

    # Mean over all outer seeds
    if return_averaged:
        fin_pearson_train = np.mean(list_all_seed_pearson_train)
        fin_pearson_train_p = np.mean(list_all_seed_pearson_p_train)
        std_pearson_train = np.std(list_all_seed_pearson_train)
        fin_spearman_train = np.mean(list_all_seed_spearman_train)
        fin_spearman_train_p = np.mean(list_all_seed_spearman_p_train)
        std_spearman_train = np.std(list_all_seed_spearman_train)
        fin_mae_train = np.mean(list_all_seed_mae_train)
        std_mae_train = np.std(list_all_seed_mae_train)
        fin_std_ratio_train = np.mean(list_all_seed_std_ratio_train)
        std_std_ratio_train = np.std(list_all_seed_std_ratio_train)

        fin_pearson_test = np.mean(list_all_seed_pearson_test)
        fin_pearson_test_p = np.mean(list_all_seed_pearson_p_test)
        std_pearson_test = np.std(list_all_seed_pearson_test)
        fin_spearman_test = np.mean(list_all_seed_spearman_test)
        fin_spearman_test_p = np.mean(list_all_seed_spearman_p_test)
        std_spearman_test = np.std(list_all_seed_spearman_test)
        fin_mae_test = np.mean(list_all_seed_mae_test)
        std_mae_test = np.std(list_all_seed_mae_test)
        fin_std_ratio_test = np.mean(list_all_seed_std_ratio_test)
        std_std_ratio_test = np.std(list_all_seed_std_ratio_test)
    
        train_metrics = [fin_pearson_train, fin_pearson_train_p, std_pearson_train, 
                        fin_spearman_train, fin_spearman_train_p, std_spearman_train, 
                        fin_mae_train, std_mae_train, fin_std_ratio_train, std_std_ratio_train]
        test_metrics = [fin_pearson_test, fin_pearson_test_p, std_pearson_test, 
                        fin_spearman_test, fin_spearman_test_p, std_spearman_test, 
                        fin_mae_test, std_mae_test, fin_std_ratio_test, std_std_ratio_test]
    
        return [round(x,3) for x in train_metrics], [round(x,3) for x in test_metrics], list_all_seed_predictions
    
    else: 
        train_metrics = [list_all_seed_pearson_train, [], [], list_all_seed_spearman_train, [], [],
                        list_all_seed_mae_train, [], list_all_seed_std_ratio_train, []]
        test_metrics = [list_all_seed_pearson_test, [], [], list_all_seed_spearman_test, [], [],
                        list_all_seed_mae_test, [], list_all_seed_std_ratio_test, []]
    
        return train_metrics, test_metrics, list_all_seed_predictions
    
     

