"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

import numpy as np
import scipy.stats
from sklearn.metrics import mean_absolute_error


def perf_metrix(predictions: list, true_values: list):
    '''Compute various prediction performances metrics.'''
    
    pearson, pearson_p = scipy.stats.pearsonr(predictions, true_values)
    spearman, spearman_p = scipy.stats.spearmanr(predictions, true_values)
    mae = mean_absolute_error(predictions, true_values)
    
    pearson = float("{0:.3g}".format(pearson))
    pearson_p = float("{0:.3g}".format(pearson_p))
    spearman = float("{0:.3g}".format(spearman))
    spearman_p = float("{0:.3g}".format(spearman_p))
    mae = float("{0:.3g}".format(mae))
    std_ratio = float("{0:.3g}".format(np.std(predictions)/np.std(true_values)))
  
    return pearson, pearson_p, spearman, spearman_p, mae, std_ratio


def is_protein(seq, aa_list):
    """
    Check if a str corresponds to a protein sequence
    return bool
    """
    for aa in seq:
        if aa not in aa_list:
            return False
    return True
