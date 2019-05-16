"""
================================================================
     Copyright (c) 2018, Yale University
                     All Rights Reserved
================================================================

 NAME : ji_utils_ml2018.py
 @DATE     Created: 11/20/18 (5:19 PM)
	       Modifications:
	       - 11/20/18: added clustering for trades

 @AUTHOR          : Jaime Shinsuke Ide
                    jaime.ide@yale.edu
===============================================================
"""


## General import
import sys
import os
import pandas as pd
from IPython.display import display
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import time

## Clustering
import mdp
import difflib
import numpy as np
import sklearn.cluster
import distance

## Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

## Regression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pprint import pprint # print Sklearn model parameters
import xgboost as xgb
import torch

class matlablike():
    pass

############################################################################
## Feature Engineering
############################################################################

def run_df_normalization(df,normType='std'):
    if normType=='std':
        print('- All columns standardized: (x-mean)/std ...')
        scaled_features = StandardScaler().fit_transform(df)
        #scaled_target_values = StandardScaler().fit_transform(target_values.reshape(-1, 1)) # Not working properly...
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    if normType=='quant':
        print('- All columns Quantile Normalizer(output:Gaussian) ...')
        scaled_features = QuantileTransformer().fit_transform(df)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return df

def remove_outliers_df_up(dfx,percent,vars2rem,showplot=True):
    df = dfx.copy()
    print('Before:',df.shape)
    for i in vars2rem:
        t = np.percentile(df[i],100-percent)
        mymax = np.max(df[i])
        # remove
        y2exclude = np.abs(df[i].values)>t
        df = df.loc[~y2exclude]
        t2add = 'excluded %d outliers (max=%2.4f)'%(y2exclude.sum(),mymax)
        # plot
        if showplot:
            sns.distplot(df[i])
            plt.plot([t, t], [0,1])
            plt.text(t-1.1, .5,'%s%% treshold:%2.2f'%(percent,t), bbox=dict(facecolor='red', alpha=0.5))
            plt.title('Removing outliers: '+ i +' (%s)'%(t2add))
            plt.show()
    print('After:',df.shape)
    return df

def remove_outliers_df_down(dfx,percent,vars2rem,showplot=True):
    df = dfx.copy()
    print('Before:',df.shape)
    for i in vars2rem:
        t = np.percentile(df[i],percent)
        mymin = np.min(df[i])
        # remove
        y2exclude = np.abs(df[i].values)<t
        df = df.loc[~y2exclude]
        t2add = 'excluded %d outliers (min=%2.4f)'%(y2exclude.sum(),mymin)
        if showplot:
            # plot
            sns.distplot(df[i])
            plt.plot([t, t], [0,1])
            plt.text(t-1.1, .5,'%s%% treshold:%2.2f'%(percent,t), bbox=dict(facecolor='red', alpha=0.5))
            plt.title('Removing outliers: '+ i +' (%s)'%(t2add))
            plt.show()
    print('After:',df.shape)
    return df

#####################################################################################################
### SUPERVISED LEARNING #############################################################################
#####################################################################################################


## Basic linear regression using Pytorch
# Ref: https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch/notebook
    # You could enter your model as well as parameter...
    # Define the model

# Ref: good wrapper for running pytorch with sklearn pipeline
# - https://www.kaggle.com/graymant/pytorch-regression-with-sklearn-pipelines

def run_pytorch_linearReg(X, Y, stepsize=1e-5, nepochs=100):
    '''
    Simple regression:
         Y = W*X + B

    Sample data to try:
    # Input (temp, rainfall, humidity)
    inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')
     # Targets (apples, oranges)
    targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')
                   '''

    def model(x):
        return x @ w.t() + b  # @ is for matrix multiplication: [5x3] x [3,2]

    # MSE loss
    def mse(t1, t2):
        diff = t1 - t2
        return torch.sum(diff * diff) / diff.numel()

    print('- Running linear regression with Pytorch...')
    # Convert inputs and targets to tensors
    inputs = torch.from_numpy(X)
    targets = torch.from_numpy(Y)
    print('X:', inputs)
    print('y:', targets)
    # Weights and biases
    nreg = len(inputs[0])
    nout = len(targets[0])
    w = torch.randn(nout, nreg, requires_grad=True)
    b = torch.randn(nout, requires_grad=True)
    # print(w)
    # print(b)

    # Generate predictions
    preds = model(inputs)
    print('- Initial prediction:',preds)

    # Training steps
    # 1. Predict
    # 2. Calculate loss
    # 3. Compute gradients with respect to parameters, i.e. w and b
    # 4. Update the parameters based on the gradients (gradient descent)
    # 5. Reset the gradient to zero

    # Train for 100 epochs
    for i in range(nepochs):
        preds = model(inputs)
        loss = mse(preds, targets)
        loss.backward() # compute gradient
        with torch.no_grad(): # no_grad will not do gradient operations and will speed-up!!
            w -= w.grad * stepsize # Think on the curve of a loss function (concave down). If gradient is positive, we have to go left, decrease w, and vice-versa
            b -= b.grad * stepsize
            w.grad.zero_() # Has to reset gradient! Default is to do cumulative
            b.grad.zero_()
        if i%10==0:
            print('-- Epoch %d: Loss=%2.4f...'%(i,loss))

    # Calculate final loss
    preds = model(inputs)
    loss = mse(preds, targets)
    print('- Final loss (MSE):',loss)
    print('- Final prediction:',preds)


    return preds,w, b


#####################################################################################################
### SUPERVISED LEARNING #############################################################################
#####################################################################################################

## XGBoost with gridSearch
def run_gridSearch_XGBoost_reg(X,y,options):
    tic = time.time()

    # 1) Parameter Tuning
    model = xgb.XGBRegressor()

    #params = {'min_child_weight': [4, 5], 'gamma': [i / 10.0 for i in range(3, 6)],
    #          'subsample': [i / 10.0 for i in range(6, 11)],
    #          'colsample_bytree': [i / 10.0 for i in range(6, 11)], 'max_depth': [2, 3, 4]}

    if options.set2use == 0: # 48 combinations
        param_dist = {"max_depth": [5,7],  # keep simple to avoid overfitting
                      "min_child_weight": [3, 6], # larger value avoids overfitting
                      "n_estimators": [100],
                      "gamma": [0,0.2],
                      'colsample_bytree': [0.6,0.75], # default is 1
                      "learning_rate": [0.01, 0.05]}
    else:  # 125 combinations
        param_dist = {"max_depth": [10, 30, 50, 70, 90],
                      "min_child_weight": [1, 3, 6, 9, 12],
                      "n_estimators": [200],
                      "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2], }


    # Create custom cv (time series)
    ii = np.arange(100)
    mys = np.array_split(ii, 3)
    print('- Gridsearch CV done by time...')
    mycv = [(np.concatenate((mys[0], mys[1]), axis=None), mys[2]),
            (np.concatenate((mys[0], mys[2]), axis=None), mys[1]),
            (np.concatenate((mys[1], mys[2]), axis=None), mys[0])]

    # Grid search
    grid_search = GridSearchCV(model, param_grid=param_dist, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    toc = time.time()
    print("- Computation time (parameter tunning): %d seconds..." % (toc - tic))
    best = grid_search.best_params_
    print('Best XGB parameters:', best)

    # 3) Run with best parameters
    tic = time.time()
    model = xgb.XGBRegressor(max_depth=best['max_depth'], min_child_weight=best['min_child_weight'],
                              n_estimators=best['n_estimators'], n_jobs=-1, verbose=1,
                              colsample_bytree=best['colsample_bytree'],gamma=best['gamma'],
                              learning_rate=best['learning_rate'])
    #model.fit(X,y)
    return model

## Random Search
# Ref: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
def run_randomSearch_RF_regressor(X,y):
    """Random search
    - It is easy to overfit...
    """

    # 1) Set search range
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # too much...
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=5)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    max_features = ['auto'] # keep default...
    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num=11)] # too much for our data...
    max_depth = [int(x) for x in np.linspace(10, 30, num=3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    min_samples_leaf = [1, 2]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    bootstrap = [True]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print('- Random search range:')
    pprint(random_grid)

    # 2) Fit
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    print('- Gridsearch CV done permutation (not for time-series)...')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=20, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)  # NEED DATA...
    # Get the best parameters
    print('- Best RF parameters found:',rf_random.best_params_)

    return rf_random


def run_gridSearch_RF_regressor(X, y):
    """Grid search
    - It is easy to overfit...
    """

    # 1) Set search range
    set2use = 1
    if set2use==0:
        # Create the parameter grid based on the results of random search
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
    if set2use == 1: # Simple model
        # Create the parameter grid based on the results of random search
        param_grid = {
            'bootstrap': [True],
            'max_depth': [3,5,7],
            'max_features': ['auto'],
            'min_samples_leaf': [1,3],
            'min_samples_split': [2,5],
            'n_estimators': [100]
        }

    print('- Grid search range:')
    pprint(param_grid)

    # 2) Fit
    # Create a based model
    rf = RandomForestRegressor()
    # Create custom cv (time series)
    ii = np.arange(100)
    mys = np.array_split(ii, 3)
    print('- Gridsearch CV done by time...')
    mycv = [(np.concatenate((mys[0], mys[1]), axis=None), mys[2]),
            (np.concatenate((mys[0], mys[2]), axis=None), mys[1]),
            (np.concatenate((mys[1], mys[2]), axis=None), mys[0])]

    # Instantiate the grid search model
    rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=mycv, n_jobs=-1, verbose=1) # '-1' is to use all processors... :P

    # Fit the random search model
    rf_grid.fit(X, y)  # NEED DATA...
    # Get the best parameters
    print('- Best RF parameters found:', rf_grid.best_params_)

    return rf_grid


def wrapper_run_regression_models(methods, data_train, data_test, yesplot=False, gridSearch=True):
    """Run regressions given train and test data.
    - Linear model (lm)
    - Ridge regression (ridge)
    - Random Forest (rf)
    - XGBoost
    - Sample use:
        data_train.X: DataFrame, with features at columns
        data_train.X_labels: list of String, with column names
        data_train.y: np array, with target value
        data_train.y_label: String, with target name
        data_train.options.label = 'custom label'
        (same for test)
    """

    options = matlablike()

    print('---------------------------------------------------- ')
    print('----- Running regression models (%s) ----- ' % (data_train.options.label))
    print('---------------------------------------------------- ')
    # myseed = [] # no seed
    myseed = 101  # fix seed

    create_split = False
    if create_split:
        X_train, X_test, y_train, y_test = train_test_split(data_train.X, data_train.y, test_size=0.3, random_state=101)
    else:
        X_train, X_test, y_train, y_test = data_train.X, data_test.X, data_train.y, data_test.y

    # 1) Shuffle and organize Train and Test data sets
    if myseed:
        itrain = np.random.RandomState(seed=myseed).permutation(len(X_train))
        itest = np.random.RandomState(seed=myseed).permutation(len(X_test))
    else:
        itrain = np.random.permutation(len(X_train))
        itest = np.random.permutation(len(X_test))

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True).iloc[itrain], \
                                       X_test.reset_index(drop=True).iloc[itest], \
                                       y_train[itrain], y_test[itest]

    print('Data shapes (train and test):', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # model_svr_lin = SVR(kernel='linear', C=1e3) # Too slow... for this size
    # model_svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    dfres_all = pd.DataFrame(columns=['Ntrain','Ntest'])
    dfres_all.loc[0,'Ntrain'] = len(itrain)
    dfres_all.loc[0,'Ntest'] = len(itest)


    mycols = ['RMSE_Train','R2_Train','RMSE_Test','R2_Test','featW']
    for method in methods:
        #dfres = dataFrame(columns=[s+'_'+method for s in mycols])
        dfres = pd.DataFrame(columns=mycols)
        if gridSearch:
            modelX = None
            if method == 'lm':
                # 2) Define and fit model
                modelX = LinearRegression()

            if method == 'ridge':
                modelX = Ridge(alpha=0.1)  # alpha corresponds to C^-1

            if method == 'xgb':
                # XGB
                options.set2use = 0  # basic set
                modelX = run_gridSearch_XGBoost_reg(X_train, y_train, options)

            if method == 'rf':
                # RF
                # model_rf = run_randomSearch_RF_regressor(X_train, y_train)
                rf_grid = run_gridSearch_RF_regressor(X_train, y_train)
                # Seet manually for now... (TODO: find a way to do automatically... the problem is grid search model does not return feature importance)
                d = rf_grid.best_params_
                modelX = RandomForestRegressor(n_estimators=d['n_estimators'], min_samples_split=d['min_samples_split'],
                                                 min_samples_leaf=d['min_samples_leaf'],
                                                 max_features=d['max_features'], max_depth=d['max_depth'])
        else:
            # Ad hoc model...
            if method == 'lm':
                # 2) Define and fit model
                modelX = LinearRegression()

            if method == 'ridge':
                modelX = Ridge(alpha=0.1)  # alpha corresponds to C^-1

            if method == 'rf':
                # XGB
                modelX = RandomForestRegressor(max_depth=3, random_state=101, n_estimators=100)

            if method == 'xgb':
                modelX = xgb.XGBRegressor(max_depth=3, n_estimators=100, n_jobs=-1)


        # model_svr_lin.fit(X_train,y_train)
        # model_svr_poly.fit(X_train,y_train)
        modelX.fit(X_train, y_train)

        # 3) Predict
        #   a) Train
        y_predx = modelX.predict(X_train)
        #   b) Test
        y_pred = modelX.predict(X_test)

        # 4) Evaluate:RMSE and plot
        a1 = np.sqrt(metrics.mean_squared_error(y_train, y_predx))
        a4 = metrics.r2_score(y_train, y_predx)
        print('-- Train --')
        print('RMSE (train) =', a1)
        print('R-squared (train):', a4)
        a7 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        a10 = metrics.r2_score(y_test, y_pred)

        print('-- Test --')
        print('RMSE (test) =', a7)
        print('R-squared (test):', a10)

        # 5) [x] 2. Checking stats with the stats regression and compare with the computed one.
        # X2 = sm.add_constant(X_train)
        # est = sm.OLS(y_train, X2)
        # est2 = est.fit()
        # print(est2.summary())

        # 6) Plot predictions
        if yesplot:
            aux = np.array([y_test, y_pred])
            df_pred = pd.DataFrame(aux.T, columns=['observed', 'model'])
            #plt.scatter(y_test, y_pred, alpha=0.5)
            j = sns.jointplot(y_test, y_pred, alpha=0.5)
            sns.lmplot(x="observed", y="model", data=df_pred, x_estimator=np.mean)

            #    j = sns.jointplot(x="orig",y="lm",data=df_pred,kind='reg')
            j.annotate(stats.pearsonr)
            # plt.legend(['orig','lm','RF'])
            plt.ylabel('predicted')
            plt.xlabel('original')
            plt.title('Regression')
            plt.show()

        # 7) Examine coefficients
        if (method == 'lm')| (method == 'ridge'):
            featWs = modelX.coef_
        if (method == 'rf') | (method == 'xgb'):
            featWs = modelX.feature_importances_

        # print('-- Beta or VarImportance(RF) --')
        # print('Model \t SurpriseCumu \t Surprise30Min \tSurpriseCumu.PrevDay')
        # print('Linear \t %2.4f \t %2.4f \t %2.4f' % (coef_lm[0], coef_lm[1], coef_lm[2]))
        # print('Ridge \t %2.4f \t %2.4f \t %2.4f' % (coef_ridge[0], coef_ridge[1], coef_ridge[2]))
        # print('RF_reg \t %2.4f \t %2.4f \t %2.4f' % (vim_rf[0], vim_rf[1], vim_rf[2]))

        # 8) Store to dataframe
        # dfres.etype = data.options.EventType
        # dfres.isETF = data.options.isETF
        dfres.loc[0,'RMSE_Train'] = a1
        dfres.loc[0,'R2_Train'] = a4
        dfres.loc[0,'RMSE_Test'] = a7
        dfres.loc[0,'R2_Test'] = a10
        dfres.loc[0,'featW'] = ' '.join(str(format(i, '.4f')) for i in featWs)
        # Add method to column name
        dfres.columns = dfres.columns+'_'+method
        dfres_all = pd.concat([dfres_all,dfres],axis=1)

    return dfres_all


def helper_run_lm_ridge_rf_xgb(dfres_all, data_train, data_test, yesplot=False, gridSearch = True):
    """Run regressions given train and test data.
    - Linear model (lm)
    - Ridge regression (ridge)
    - Random Forest (rf)
    - XGBoost
    - Sample use:
        dfres_all = pd.DataFrame(columns=['perc','LG','bin',
            'Ntrain','RMSE_lin','RMSE_rid','RMSE_RF','RMSE_XGB',
            'R2_lin','R2_rid','R2_RF','R2_XGB',
            'Ntest','RMSE_lin_Test','RMSE_rid_Test','RMSE_RF_Test','RMSE_XGB_Test',
            'R2_lin_Test','R2_rid_Test','R2_RF_Test','R2_XGB_Test',
            'coef_lin','coef_ridge','vim_rf','vim_xgb'])

    """


    data = data_test
    options = matlablike()

    print('---------------------------------------------------- ')
    print('----- Running routine 3 (perc=%1.2f LG=%d Bin=%d) ----- ' % (
    data.options.outlier_percent, data.options.LG[0], data.options.bin[0]))
    print('---------------------------------------------------- ')
    # myseed = [] # no seed
    myseed = 101  # fix seed

    create_split = False
    if create_split:
        X_train, X_test, y_train, y_test = train_test_split(data_train.X,data_train.y,test_size=0.3,random_state=101)
    else:
        X_train, X_test, y_train, y_test = data_train.X, data_test.X, data_train.y, data_test.y

    # 1) Shuffle and organize Train and Test data sets
    if myseed:
        itrain = np.random.RandomState(seed=myseed).permutation(len(X_train))
        itest = np.random.RandomState(seed=myseed).permutation(len(X_test))
    else:
        itrain = np.random.permutation(len(X_train))
        itest = np.random.permutation(len(X_test))
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True).iloc[itrain], \
                                       X_test.reset_index(drop=True).iloc[itest], \
                                       y_train[itrain], y_test[itest]

    print('Data shapes (train and test):', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 2) Define and fit model
    model_lm = LinearRegression()
    # model_svr_lin = SVR(kernel='linear', C=1e3) # Too slow... for this size
    # model_svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    model_ridge = Ridge(alpha=0.1)  # alpha corresponds to C^-1
    if gridSearch:
        # XGB
        options.set2use = 0  # basic set
        model_xgb = run_gridSearch_XGBoost_reg(X_train, y_train, options)

        # RF
        #model_rf = run_randomSearch_RF_regressor(X_train, y_train)
        rf_grid = run_gridSearch_RF_regressor(X_train, y_train)
        # Seet manually for now... (TODO: find a way to do automatically... the problem is grid search model does not return feature importance)
        d = rf_grid.best_params_
        model_rf = RandomForestRegressor(n_estimators=d['n_estimators'], min_samples_split = d['min_samples_split'],min_samples_leaf=d['min_samples_leaf'],
                        max_features = d['max_features'],max_depth = d['max_depth'])
    else:
        # Ad hoc model...
        model_rf = RandomForestRegressor(max_depth=3, random_state=101, n_estimators=100)
        model_xgb = xgb.XGBRegressor(max_depth=3,n_estimators=100, n_jobs=-1)

    model_lm.fit(X_train, y_train)
    # model_svr_lin.fit(X_train,y_train)
    # model_svr_poly.fit(X_train,y_train)
    model_ridge.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)

    # 3) Predict
    #   a) Train
    y_predx = model_lm.predict(X_train)
    # y_pred_svr_lin = model_svr_lin.predict(X_test)
    # y_pred_svr_poly = model_svr_poly.predict(X_test)
    y_pred_ridgex = model_ridge.predict(X_train)
    y_pred_rfx = model_rf.predict(X_train)
    y_pred_xgbx = model_xgb.predict(X_train)

    #   b) Test
    y_pred = model_lm.predict(X_test)
    # y_pred_svr_lin = model_svr_lin.predict(X_test)
    # y_pred_svr_poly = model_svr_poly.predict(X_test)
    y_pred_ridge = model_ridge.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_xgb = model_xgb.predict(X_test)

    # 4) Evaluate:RMSE and plot
    a1 = np.sqrt(metrics.mean_squared_error(y_train, y_predx))
    a2 = np.sqrt(metrics.mean_squared_error(y_train, y_pred_ridgex))
    a3 = np.sqrt(metrics.mean_squared_error(y_train, y_pred_rfx))
    a31 = np.sqrt(metrics.mean_squared_error(y_train, y_pred_xgbx))
    a4 = metrics.r2_score(y_train, y_predx)
    a5 = metrics.r2_score(y_train, y_pred_ridgex)
    a6 = metrics.r2_score(y_train, y_pred_rfx)
    a61 = metrics.r2_score(y_train, y_pred_xgbx)
    print('-- Train --')
    print('RMSE_linReg (train) =', a1)
    print('RMSE_Ridge (train) =', a2)
    print('RMSE_RF (train) =', a3)
    print('RMSE_XGB (train) =', a31)
    print('R-squared_linReg (train):', a4)
    print('R-squared_Ridge (train):', a5)
    print('R-squared_RF (train):', a6)
    print('R-squared_XGB (train):', a61)
    a7 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    a8 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))
    a9 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
    a91 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))

    a10 = metrics.r2_score(y_test, y_pred)
    a11 = metrics.r2_score(y_test, y_pred_ridge)
    a12 = metrics.r2_score(y_test, y_pred_rf)
    a121 = metrics.r2_score(y_test, y_pred_xgb)

    print('-- Test --')
    print('RMSE_linReg (test) =', a7)
    print('RMSE_Ridge (test) =', a8)
    print('RMSE_RF (test) =', a9)
    print('RMSE_XGB (test) =', a91)
    print('R-squared_linReg (test):', a10)
    print('R-squared_Ridge (test):', a11)
    print('R-squared_RF (test):', a12)
    print('R-squared_XGB (test):', a121)

    # 5) [x] 2. Checking stats with the stats regression and compare with the computed one.
    # X2 = sm.add_constant(X_train)
    # est = sm.OLS(y_train, X2)
    # est2 = est.fit()
    # print(est2.summary())

    # 6) Plot predictions
    if yesplot:
        aux = np.array([y_test,y_pred,y_pred_rf])
        df_pred = pd.DataFrame(aux.T,columns=['orig','lm','RF'])
        plt.scatter(y_test,y_pred,alpha=0.5)
        j = sns.jointplot(y_test,y_pred,alpha=0.5)
        sns.lmplot(x="orig",y="lm",data=df_pred,x_estimator=np.mean)

    #    j = sns.jointplot(x="orig",y="lm",data=df_pred,kind='reg')
        j.annotate(stats.pearsonr)
        #plt.legend(['orig','lm','RF'])
        plt.ylabel('predicted')
        plt.xlabel('original')
        plt.title('Regression')
        plt.show()

    # 7) Examine coefficients
    coef_lm = model_lm.coef_
    coef_ridge = model_ridge.coef_
    vim_rf = model_rf.feature_importances_
    vim_xgb = model_xgb.feature_importances_

    #print('-- Beta or VarImportance(RF) --')
    #print('Model \t SurpriseCumu \t Surprise30Min \tSurpriseCumu.PrevDay')
    #print('Linear \t %2.4f \t %2.4f \t %2.4f' % (coef_lm[0], coef_lm[1], coef_lm[2]))
    #print('Ridge \t %2.4f \t %2.4f \t %2.4f' % (coef_ridge[0], coef_ridge[1], coef_ridge[2]))
    #print('RF_reg \t %2.4f \t %2.4f \t %2.4f' % (vim_rf[0], vim_rf[1], vim_rf[2]))

    # 8) Store to dataframe
    dfres = pd.DataFrame(columns=dfres_all.columns)
    #dfres.etype = data.options.EventType
    #dfres.isETF = data.options.isETF
    dfres.perc = data.options.outlier_percent
    dfres.LG = data.options.LG
    dfres.bin = data.options.bin
    dfres.Ntrain = len(itrain)
    dfres.Ntest = len(itest)
    dfres.RMSE_lin = a1
    dfres.RMSE_rid = a2
    dfres.RMSE_RF = a3
    dfres.RMSE_XGB = a31
    dfres.R2_lin = a4
    dfres.R2_rid = a5
    dfres.R2_RF = a6
    dfres.R2_XGB = a61
    dfres.RMSE_lin_Test= a7
    dfres.RMSE_rid_Test = a8
    dfres.RMSE_RF_Test = a9
    dfres.RMSE_XGB_Test = a91
    dfres.R2_lin_Test = a10
    dfres.R2_rid_Test = a11
    dfres.R2_RF_Test = a12
    dfres.R2_XGB_Test = a121
    dfres.coef_lin = ' '.join(str(format(i, '.4f')) for i in coef_lm)
    dfres.coef_ridge = ' '.join(str(format(i, '.4f')) for i in coef_ridge)
    dfres.vim_rf = ' '.join(str(format(i, '.4f')) for i in vim_rf)
    dfres.vim_xgb = ' '.join(str(format(i, '.4f')) for i in vim_xgb)
    #print('vim_rf',vim_rf)

    #dfres.Bcumu_lin = coef_lm[0]
    #dfres.B30min_lin = coef_lm[1]
    #dfres.Bprev_lin = coef_lm[2]
    #dfres.Bcumu_rid = coef_ridge[0]
    #dfres.B30min_rid = coef_ridge[1]
    #dfres.Bprev_rid = coef_ridge[2]
    #dfres.Bcumu_RF = vim_rf[0]
    #dfres.B30min_RF = vim_rf[1]
    #dfres.Bprev_RF = vim_rf[2]

    dfres_all = dfres_all.append(dfres)

    return dfres_all


def helper_run_lm_ridge_rf(dfres_all, data_train, data_test, yesplot=False, gridSearchRF = True):
    """Run regressions given train and test data.
    - Linear model (lm)
    - Ridge regression (ridge)
    - Random Forest (rf)"""

    data = data_test

    print('---------------------------------------------------- ')
    print('----- Running routine 2 (perc=%1.2f LG=%d Bin=%d) ----- ' % (
    data.options.outlier_percent, data.options.LG[0], data.options.bin[0]))
    print('---------------------------------------------------- ')
    # myseed = [] # no seed
    myseed = 101  # fix seed

    create_split = False
    if create_split:
        X_train, X_test, y_train, y_test = train_test_split(data_train.X,data_train.y,test_size=0.3,random_state=101)
    else:
        X_train, X_test, y_train, y_test = data_train.X, data_test.X, data_train.y, data_test.y

    # 1) Shuffle and organize Train and Test data sets
    if myseed:
        itrain = np.random.RandomState(seed=myseed).permutation(len(X_train))
        itest = np.random.RandomState(seed=myseed).permutation(len(X_test))
    else:
        itrain = np.random.permutation(len(X_train))
        itest = np.random.permutation(len(X_test))
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True).iloc[itrain], \
                                       X_test.reset_index(drop=True).iloc[itest], \
                                       y_train[itrain], y_test[itest]

    print('Data shapes (train and test):', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 2) Define and fit model
    model_lm = LinearRegression()
    # model_svr_lin = SVR(kernel='linear', C=1e3) # Too slow... for this size
    # model_svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    model_ridge = Ridge(alpha=0.1)  # alpha corresponds to C^-1
    if gridSearchRF:
        #model_rf = run_randomSearch_RF_regressor(X_train, y_train)
        rf_grid = run_gridSearch_RF_regressor(X_train, y_train)
        # Seet manually for now... (TODO: find a way to do automatically... the problem is grid search model does not return feature importance)
        d = rf_grid.best_params_
        model_rf = RandomForestRegressor(n_estimators=d['n_estimators'], min_samples_split = d['min_samples_split'],min_samples_leaf=d['min_samples_leaf'],
                        max_features = d['max_features'],max_depth = d['max_depth'])

    else:
        # Ad hoc model...
        model_rf = RandomForestRegressor(max_depth=3, random_state=101, n_estimators=100)

    model_lm.fit(X_train, y_train)
    # model_svr_lin.fit(X_train,y_train)
    # model_svr_poly.fit(X_train,y_train)
    model_ridge.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    # 3) Predict
    #   a) Train
    y_predx = model_lm.predict(X_train)
    # y_pred_svr_lin = model_svr_lin.predict(X_test)
    # y_pred_svr_poly = model_svr_poly.predict(X_test)
    y_pred_ridgex = model_ridge.predict(X_train)
    y_pred_rfx = model_rf.predict(X_train)
    #   b) Test
    y_pred = model_lm.predict(X_test)
    # y_pred_svr_lin = model_svr_lin.predict(X_test)
    # y_pred_svr_poly = model_svr_poly.predict(X_test)
    y_pred_ridge = model_ridge.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)

    # 4) Evaluate:RMSE and plot
    a1 = np.sqrt(metrics.mean_squared_error(y_train, y_predx))
    a2 = np.sqrt(metrics.mean_squared_error(y_train, y_pred_ridgex))
    a3 = np.sqrt(metrics.mean_squared_error(y_train, y_pred_rfx))
    a4 = metrics.r2_score(y_train, y_predx)
    a5 = metrics.r2_score(y_train, y_pred_ridgex)
    a6 = metrics.r2_score(y_train, y_pred_rfx)
    print('-- Train --')
    print('RMSE_linReg (train) =', a1)
    print('RMSE_Ridge (train) =', a2)
    print('RMSE_RF (train) =', a3)
    print('R-squared_linReg (train):', a4)
    print('R-squared_Ridge (train):', a5)
    print('R-squared_RF (train):', a6)
    a7 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    a8 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))
    a9 = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))
    a10 = metrics.r2_score(y_test, y_pred)
    a11 = metrics.r2_score(y_test, y_pred_ridge)
    a12 = metrics.r2_score(y_test, y_pred_rf)
    print('-- Test --')
    print('RMSE_linReg (test) =', a7)
    print('RMSE_Ridge (test) =', a8)
    print('RMSE_RF (test) =', a9)
    print('R-squared_linReg (test):', a10)
    print('R-squared_Ridge (test):', a11)
    print('R-squared_RF (test):', a12)

    # 5) [x] 2. Checking stats with the stats regression and compare with the computed one.
    # X2 = sm.add_constant(X_train)
    # est = sm.OLS(y_train, X2)
    # est2 = est.fit()
    # print(est2.summary())

    # 6) Plot predictions
    if yesplot:
        aux = np.array([y_test,y_pred,y_pred_rf])
        df_pred = pd.DataFrame(aux.T,columns=['orig','lm','RF'])
        plt.scatter(y_test,y_pred,alpha=0.5)
        j = sns.jointplot(y_test,y_pred,alpha=0.5)
        sns.lmplot(x="orig",y="lm",data=df_pred,x_estimator=np.mean)

    #    j = sns.jointplot(x="orig",y="lm",data=df_pred,kind='reg')
        j.annotate(stats.pearsonr)
        #plt.legend(['orig','lm','RF'])
        plt.ylabel('predicted')
        plt.xlabel('original')
        plt.title('Regression')
        plt.show()

    # 7) Examine coefficients
    coef_lm = model_lm.coef_
    coef_ridge = model_ridge.coef_
    vim_rf = model_rf.feature_importances_
    print('-- Beta or VarImportance(RF) --')
    print('Model \t SurpriseCumu \t Surprise30Min \tSurpriseCumu.PrevDay')
    print('Linear \t %2.4f \t %2.4f \t %2.4f' % (coef_lm[0], coef_lm[1], coef_lm[2]))
    print('Ridge \t %2.4f \t %2.4f \t %2.4f' % (coef_ridge[0], coef_ridge[1], coef_ridge[2]))
    print('RF_reg \t %2.4f \t %2.4f \t %2.4f' % (vim_rf[0], vim_rf[1], vim_rf[2]))

    # 8) Store to dataframe
    dfres = pd.DataFrame(columns=dfres_all.columns)
    #dfres.etype = data.options.EventType
    #dfres.isETF = data.options.isETF
    dfres.perc = data.options.outlier_percent
    dfres.LG = data.options.LG
    dfres.bin = data.options.bin
    dfres.Ntrain = len(itrain)
    dfres.Ntest = len(itest)
    dfres.RMSE_lin = a1
    dfres.RMSE_rid = a2
    dfres.RMSE_RF = a3
    dfres.R2_lin = a4
    dfres.R2_rid = a5
    dfres.R2_RF = a6
    dfres.RMSE_lin_Test= a7
    dfres.RMSE_rid_Test = a8
    dfres.RMSE_RF_Test = a9
    dfres.R2_lin_Test = a10
    dfres.R2_rid_Test = a11
    dfres.R2_RF_Test = a12
    dfres.coef_lin = ' '.join(str(format(i, '.4f')) for i in coef_lm)
    dfres.coef_ridge = ' '.join(str(format(i, '.4f')) for i in coef_ridge)
    dfres.vim_rf = ' '.join(str(format(i, '.4f')) for i in vim_rf)
    #print('vim_rf',vim_rf)

    #dfres.Bcumu_lin = coef_lm[0]
    #dfres.B30min_lin = coef_lm[1]
    #dfres.Bprev_lin = coef_lm[2]
    #dfres.Bcumu_rid = coef_ridge[0]
    #dfres.B30min_rid = coef_ridge[1]
    #dfres.Bprev_rid = coef_ridge[2]
    #dfres.Bcumu_RF = vim_rf[0]
    #dfres.B30min_RF = vim_rf[1]
    #dfres.Bprev_RF = vim_rf[2]

    dfres_all = dfres_all.append(dfres)

    return dfres_all


#####################################################################################################
### CLUSTERING ######################################################################################
#####################################################################################################

def ji_time_str2sec(s):
    """Convert time in string format hh:mm:ss into seconds"""
    ftr = [3600, 60, 1]
    return sum([a * b for a, b in zip(ftr, map(int, s.split(':')))])


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_jaccard_sim_words(wd1, wd2):
    a = set(list(wd1))
    b = set(list(wd2))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def ji_seqmatch(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()




def ji_nlp_get_word_groups(words, damping, showplot):
    """Given a list of words, it returns the words by similarity"""
    # Ref: https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

    # words = "car duck school har bar put crack".split(" ") #Replace this line
    # words = np.asarray(words) #So that indexing with a list will work

    similarity = np.array([[ji_seqmatch(w1, w2) for w1 in words] for w2 in words])
    if showplot:
        plt.imshow(similarity)
        plt.title('Words similarity by SeqMatching')
        istest = False
        if istest:
            for cluster_id in np.unique(affprop.labels_):
                exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
                cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
                cluster_str = ", ".join(cluster)
                print(" - *%s:* %s" % (exemplar, cluster_str))

    # similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=damping)
    affprop.fit(similarity)

    return affprop.labels_, affprop


def ji_nlp_word2asc(word, n=5):
    """Convert word to number. Basically, as more sequencial matching, the numbers will be closer.
    Example:
    'ACSA' is closer to 'ACSB', than
    'ACSA' to 'ACFA'
    (*) The words on the left matters more!
    Sample use:
    print(ji_nlp_word2asc('A',n=1))
    print(ji_nlp_word2asc('AS',n=2))
    print(ji_nlp_word2asc('ASCB',n=4))
    65
    6583
    65836766
    """
    i0 = 3  # ignore the first 3 digits
    wd = word[i0:i0 + n]
    mynum = ''
    for i in wd:
        mynum += str(ord(i))  #
    return int(mynum)


## APPROACH 1: Cluster the orderIds

def ji_cluster_orderId(df_trades, cccy, cdate, alltime, wstart=3, wend=8, damping=0.8, showplot=1):
    """Clustering by orderId. It will use sequence matching of the characters.
    - Affinity propagation is used since it is appropriate for finding the 'exemplar' Ids, the radical of the IDs.
      It finds the examplar IDs and damp the similar ones by message passing.
    - It doesn't seem to work consistently for small samples (say n<50 trades) """
    res = matlab_like()

    # alltime = df_trades.orderstart.apply(ji_time_str2sec).values # Apply outside for efficiency
    allorderId = df_trades.orderid.values

    # Particular currency and date
    mask = (df_trades.ccy == cccy) & (df_trades.trdate == cdate)
    masked_allorders = allorderId[mask]
    corderIds = [s[wstart:wend] for s in masked_allorders]  # trim words
    ctime = alltime[mask]
    ordergroups, affprop = ji_nlp_get_word_groups(corderIds, damping, showplot)
    # Store
    res.labels = ordergroups
    res.labels_info = 'Clusters are basically grouped by orderId using affinity propagation'
    res.model_affprop = affprop

    if showplot:
        print('******************* Affinity Propagation using feats: orderIdNum ********************')
        # Plot
        mycolors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ff0088', '#ff8800', '#0088ff']
        plt.figure(figsize=(20, 10))
        for c in ordergroups:
            cctime = ctime[ordergroups == c]
            l = plt.plot(cctime / 3600, np.ones(len(cctime)) * c, 'ko')
            plt.setp(l, 'markersize', 15)
            plt.setp(l, 'markerfacecolor', mycolors[c % 8], 'alpha', 0.5)
        plt.xlabel('time(hours)', size=20)
        plt.ylabel('orderId Group (1:1 color=cluster)', size=20)
        plt.title('%s: %s (%d trades)' % (cccy, cdate, len(ordergroups)), size=30)
        plt.show()

        # words = masked_allorders
        words = corderIds
        words = np.asarray(words)  # So that indexing with a list will work
        words = np.asarray(words)  # So that indexing with a list will work
        print('** Exemplar samples and members of the clan **')
        for i, cluster_id in enumerate(np.unique(affprop.labels_)):
            exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            # cluster_str = ".., ".join(cluster)
            # print(" - *%s:* %s" % (exemplar,cluster_str))
            print(" - * %s (cluster#%d):" % (exemplar, i))
            for s in cluster:
                print('     %s' % (s))

        print('******************')
        for i, j, k in zip(corderId, ordergroups, ctime):
            print('%s: %d (%1.1f hours)' % (i, j, k / 3600))
        print('****************** (END) ********************** \n')

    return res


## APPROACH 2: Convert the orderId into number and use as feature in the clustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def ji_cluster_dbscan(df_trades, cccy, cdate, alltime, allorderIdNum, wstart=3, wend=8, max_dist=0.25, showplot=1,
                      showtext=0):
    """Clustering by time and orderIdNum(converted to AscII). Converting to AcsII will make to clusters that have similar
       IDs on the left-side.
    - Density-Based Spatial Clustering and Application with Noise (DBSCAN) is used since it is appropriate for finding
       dense trades clusters. Clan is defined around the center based on distance btwn samples. So it will cluster samples
       that are close each other.
    - It seems to work well with small and large sample sizes, thus more robust. """
    res = matlab_like()

    # alltime = df_trades.orderstart.apply(ji_time_str2sec).values # Apply outside once for efficiency
    allorderId = df_trades.orderid.values
    # allorderIdNum = df_trades.orderid.apply(ji_nlp_word2asc,n=wend-wstart+1).values

    # Particular currency and date
    mask = (df_trades.ccy == cccy) & (df_trades.trdate == cdate)
    masked_orders = allorderIdNum[mask]
    corderIds = [s[wstart:wend] for s in allorderId[mask]]  # for visualization

    masked_time = alltime[mask]
    X = [[i, j] for i, j in zip(masked_time, masked_orders)]  # seconds, Asc(orderId)

    # #############################################################################
    # Generate sample data
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
    #                            random_state=0)

    # Standardize
    do_scale = 1
    X0 = np.asarray(X)
    if do_scale:
        X = StandardScaler().fit_transform(X)
    else:
        X = np.asarray(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=max_dist, min_samples=3).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Store
    res.labels = labels
    res.label_info = '-1 is for the noise, not counted as cluster'
    res.n_clusters = n_clusters_
    res.db = db

    if showplot:
        print('******************* DBSCAN using feats: time, orderIdNum ********************')
        # #############################################################################
        # Plot result

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each) # Sample colormaps: 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
        #                               'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        colors = [plt.cm.RdYlBu(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        plt.figure(figsize=(20, 10))
        cont = 0
        mycolors = ['#ff2ff0', '#ff880f', '#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff', '#ff0088', '#ff8800',
                    '#0088ff']
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            # k>=0) standard clusters
            xy = X[class_member_mask & core_samples_mask]
            xy0 = X0[class_member_mask & core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #         markeredgecolor='k', markersize=14)

            if showtext:
                plt.plot(xy0[:, 0] / 3600, xy[:, 1], '.', markerfacecolor=mycolors[cont % 10],
                         markeredgecolor='y', markersize=1)
                # add annotation
                # print('k:',k,xy0[:, 0]/3600, xy[:, 1])
                for cx, cy in zip(xy0[:, 0] / 3600, xy[:, 1]):
                    plt.annotate(k, (cx, cy),
                                 horizontalalignment='center', verticalalignment='center', fontsize=20,
                                 color=mycolors[cont % 10])
            else:
                plt.plot(xy0[:, 0] / 3600, xy[:, 1], 'o', markerfacecolor=mycolors[cont % 10],
                         markeredgecolor='k', markersize=14)

            # -1) noise clusters
            xy = X[class_member_mask & ~core_samples_mask]
            xy0 = X0[class_member_mask & ~core_samples_mask]
            plt.plot(xy0[:, 0] / 3600, xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=mycolors[cont%8],
            #         markeredgecolor='k', markersize=6)
            cont += 1

        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.xlabel('time(hours)', size=20)
        plt.ylabel('orderId Group (ascII)', size=20)
        plt.title(
            '%s: %s (%d trades, %d clusters) - DBSCAN(time,orderId)' % (cccy, cdate, len(masked_time), n_clusters_),
            size=20)
        plt.show()

        print('******************')
        for i, j, k in zip(corderId, labels, masked_time):
            print('%s: %d (%1.1f hours)' % (i, j, k / 3600))
        print('****************** (END) ********************** \n')

    return res