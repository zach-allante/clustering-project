import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 
import scipy.stats as stats
import math
import sklearn.preprocessing
from env import get_db_url

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def make_location_model_df(df,theme, n_clust):
    ''' This function will make a df filled with only the location information needed to proceed into modeling'''
    Scaler = MinMaxScaler()
    df[theme] = Scaler.fit_transform(df[theme])
    new = df[theme]
    kmeans = KMeans(n_clusters= n_clust, random_state = 123)
    kmeans.fit(new)
    new['cluster'] = kmeans.predict(new)
    new[['LCluster0','LCluster1','LCluster2','LCluster3','LCluster4']] = pd.get_dummies(new['cluster'])
    new = new.drop(columns = ['LCluster0','LCluster3','LCluster4','cluster'])
    return(new)

def make_foundation_model_df(df,theme, n_clust):
    ''' this function will return a df filled with only the foundation information needed to proceed into modeling'''
    Scaler = MinMaxScaler()
    df[theme] = Scaler.fit_transform(df[theme])
    new = df[theme]
    kmeans = KMeans(n_clusters= n_clust, random_state = 123)
    kmeans.fit(new)
    new['cluster'] = kmeans.predict(new)
    new[['FCluster0','FCluster1','FCluster2','FCluster3']] = pd.get_dummies(new['cluster'])
    new = new.drop(columns = ['FCluster0','FCluster1','FCluster3','cluster'])
    return(new)

def regression_functions(modeltrain,modelvalidate,modeltest,train,validate):
    ''' This function will return the RMSE scores on train and validate for a polynomial regression model'''
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_degree2Location = pf.fit_transform(modeltrain)

    # transform X_validate_scaled & X_test_scaled
    x_validate_degree2Location = pf.transform(modelvalidate)
    x_test_degree2Location = pf.transform(modeltest)

    # create the model object
    lm2Location = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train,
    # since we have converted it to a dataframe from a series!
    lm2Location.fit(x_train_degree2Location, train.logerror)

    # predict train
    train['logerror_pred_lm2Loc'] = lm2Location.predict(x_train_degree2Location)

    # evaluate: rmse
    rmse_train = mean_squared_error(train.logerror, train.logerror_pred_lm2Loc) ** (1 / 2)

    # predict validate
    validate['logerror_pred_lm2Loc'] = lm2Location.predict(x_validate_degree2Location)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate.logerror, validate.logerror_pred_lm2Loc) ** (1 / 2)

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train,
      "\nValidation/Out-of-Sample: ", rmse_validate)
    
def Lass(modeltrain,modelvalidate,train,validate):
    ''' This function will return the rmse scores on train and validate for a polynomial regression model'''
    # Lasso Lars model using location features/clusters

    #This is where I create my Lasso Lars model
    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model

    lars.fit(modeltrain, train.logerror)

    # predict train
    train['logerror_pred_lars'] = lars.predict(modeltrain)

    # evaluate: rmse
    rmse_train = mean_squared_error(train.logerror, train.logerror_pred_lars) ** (1 / 2)

    # predict validate
    validate['logerror_pred_lars'] = lars.predict(modelvalidate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate.logerror,validate.logerror_pred_lars) ** (1 / 2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train,
      "\nValidation/Out-of-Sample: ", rmse_validate)


    
    
    
def model_test(modeltrain,modelvalidate,modeltest,validate,test):
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_degree2Location = pf.fit_transform(modeltrain)

    # transform X_validate_scaled & X_test_scaled
    x_validate_degree2Location = pf.transform(modelvalidate)
    x_test_degree2Location = pf.transform(modeltest)
    # create the model object
    lm2Location = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train,
    # since we have converted it to a dataframe from a series!
    lm2Location.fit(x_validate_degree2Location, validate.logerror)

    # predict train
    validate['logerror_pred_lm2Loc'] = lm2Location.predict(x_validate_degree2Location)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate.logerror, validate.logerror_pred_lm2Loc) ** (1 / 2)

    # predict validate
    test['logerror_pred_lm2Loc'] = lm2Location.predict(x_test_degree2Location)

    # evaluate: rmse
    rmse_test = mean_squared_error(test.logerror, test.logerror_pred_lm2Loc) ** (1 / 2)

    print("RMSE for Polynomial Model, degrees=2\nValidate/Out-of-Sample: ", rmse_validate,
      "\nTest/Out-of-Sample: ", rmse_test)
    
    
def baseline(df):  
    ''' This function will return the baseline rmse score'''
    logerror_pred_median = df['logerror'].median()
    df['logerror_pred_median'] = logerror_pred_median
# RMSE of prop_value_pred_median
    rmse_baseline_train = mean_squared_error(df.logerror, df.logerror_pred_median)**(1/2)
#printing results of baseline model
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_baseline_train, 2),)