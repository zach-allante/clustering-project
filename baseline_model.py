import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
import sklearn.preprocessing


def model_columns(train,validate,test):
    '''This function will provide my models with the correct features to run for their x and y values'''

    # assigning features to be used for modeling  
    x_cols = ['baths','rooms_count','sqft']
    y_train = train['property_value']

    # changing y train into a dataframe to append the new column with predicted values 
    y_train = pd.DataFrame(y_train)

    # assigning features to be used for modeling to train, validate and test data sets
    X_train = train[x_cols]
    X_validate = validate[x_cols]
    y_validate = validate['property_value']
    X_test = test[x_cols]
    y_test = test['property_value']

    # changing y validate into a dataframe to append the new column with predicted values 
    y_validate= pd.DataFrame(y_validate)

    # changing y validate into a dataframe to append the new column with predicted values
    y_test= pd.DataFrame(y_test)

    return X_train, X_validate, y_train, y_validate, X_test, y_test


def scaling(X_train,X_validate,X_test):
    '''function will scale features across data sets train, validate and test'''
    # applying scaling to all the data splits.
    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(X_train)

    # transforming train, validate and test datasets
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
  
    return X_train_scaled, X_validate_scaled, X_test_scaled


def baseline_model(y_train,y_validate):
    '''this function will create my baseline model'''

    # Predict property_value_pred_mean (not used in project)
    prop_value_pred_mean = y_train['property_value'].mean()
    y_train['prop_value_pred_mean'] = prop_value_pred_mean
    y_validate['prop_value_pred_mean'] = prop_value_pred_mean

    # compute prop_value_pred_median
    prop_value_pred_median = y_train['property_value'].median()
    y_train['prop_value_pred_median'] = prop_value_pred_median
    y_validate['prop_value_pred_median'] = prop_value_pred_median

    # RMSE of prop_value_pred_median
    rmse_baseline_train = mean_squared_error(y_train.property_value, y_train.prop_value_pred_median)**(1/2)
    rmse_baseline_validate = mean_squared_error(y_validate.property_value, y_validate.prop_value_pred_median)**(1/2)

    #printing results of baseline model
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_baseline_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_baseline_validate, 2))



