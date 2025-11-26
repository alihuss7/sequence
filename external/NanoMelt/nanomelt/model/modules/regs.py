"""
 Copyright 2024. Aubin Ramon and Pietro Sormanni. CC BY-NC-SA 4.0
 
"""

from sklearn.linear_model import Ridge,ElasticNet,HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor



# regression models and params
elasticnet = [ElasticNet(alpha=0.5,l1_ratio=0.1,max_iter=10000,tol=0.01), 
              {'alpha':[0.01,1,10,50,100,200,300,500],
               'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}]

huber = [HuberRegressor(max_iter=50000, alpha=100, epsilon=5,tol=0.1),
         {'epsilon':[1,1.5,5,10,30,50,100,300,500],
          'alpha':[0.0001,0.01,1,10,50,100,200,300,500]}]

ridge = [Ridge(alpha=200),
         {'alpha':[0.01,1,10,50,100,200,300,500,700,1000]}]

gpr = [GaussianProcessRegressor(random_state=0, n_restarts_optimizer=20, normalize_y=True, alpha=1e-4, 
                                kernel=RBF(length_scale_bounds=(1e-20, 1e5)) + Matern(nu=2.5, length_scale_bounds=(1e-20, 1e10)) + WhiteKernel(noise_level_bounds=(1e-50, 1e10))),
        {'alpha':[0.00001,0.0001,0.001,0.01,1,10,50,100]}]

randomforest = [RandomForestRegressor(max_depth=2, random_state=0),
                {'max_depth':range(1,15),}]

supportvector = [SVR(C=10, epsilon=1e-4),
                 {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                  'C':[0.01,0.1,1,10,20,50],
                  'epsilon':[1e-1,1e-3,1e-4,1e-5,1e-6,1e-8]}]

lightgbm = [LGBMRegressor(max_depth=5,path_smooth=5,verbosity=-1),
            {'max_depth':[-1,1,5,10,15,20],
             'extra_trees':['true','false'],
             'path_smooth':[0.0,0.1,0.5,1,5,10,50]}]


# Store regressions in a dict

regression_list = ['elasticnet','RF', 'huber', 'ridge','GPR','SVR','LightGBM']

regression_dict = { 'elasticnet':elasticnet,'RF':randomforest,
                    'huber':huber,'ridge':ridge,'GPR':gpr,
                    'SVR':supportvector,'LightGBM':lightgbm}
