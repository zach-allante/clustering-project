import pandas as pd
import numpy as np
from env import get_db_url
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from sklearn.model_selection import train_test_split

# ------------------- ACQUIRING DATA -------------------#
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = ''' SELECT *
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL'''
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))

    # Save data to csv 
    filepath = Path('zillow.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index =False)
    
    return df
    
def get_zillow_data():
    '''
    This function reads in zillow data from local copy as a df
    '''
    # Reads local copy of csv 
    df = pd.read_csv('zillow.csv')

    # renaming column names to more readable format
    df = df.rename(columns = {'bedroomcnt':'beds', 'roomcnt':'total_rooms',
                              'bathroomcnt':'baths', 
                              'calculatedfinishedsquarefeet':'sqft',
                              'taxvaluedollarcnt':'property_value', 
                              'yearbuilt':'age',})
    
    return df
    
def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # Convert binary categorical variables to objects with name of location
    cleanup_fips = {"fips":{6037: 'Los Angeles CA', 6059:'Orange County CA', 6111: 'Ventura County CA'} }    
    df = df.replace(cleanup_fips)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)      
    
    return train, validate, test    

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow_data())
    
    return train, validate, test

# ------------------- CLEANING DATA -------------------#

def nulls_by_col(df):
    '''this function will output the number and percent of nulls by column in the given df'''
    #number of rows missing in dataframe 
    num_missing = df.isnull().sum()
    #total rows in dataframe
    rows = df.shape[0]
    #defining perecent missing variable 
    percent_missing = num_missing/ rows *100
    #creating dataframe of data above
    cols_missing = pd.DataFrame({'num_rows_missing':num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing.sort_values(by='num_rows_missing',ascending=False)

def nulls_by_row(df):
    '''this function will output the number and percent of nulls in the rows of the df'''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['parcelid', 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def remove_columns(df, cols_to_remove):
    '''this function will drop columns selected from df'''
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''this funciton will make sure to keep the given percentage of the data in each row and column'''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df

def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    '''this function will remove columns and keep the requested amount of data in the df '''
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df

def scale_zillow(df,columns):
    Scaler = Minmaxscaler()
    scaled_df = Scaler.fit_transform(df[columns])
    return scaled_df
