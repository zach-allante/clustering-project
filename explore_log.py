#imports used for explore file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from tabulate import tabulate


def bar_chart(df,column1,column2):
    #Bar chart of Log Error
    plt.title("Fips and Logerror")
    sns.barplot(x=column1, y=column2, data=df)
    mean_logerror = df.column2.mean()
    plt.axhline(mean_logerror, label="Logerror", color='red', linestyle='dotted')
    plt.xlabel('')
    plt.legend()
    plt.show()

def statistic_table(df):
    ''' This function will create a table of information '''
    #calculating median of property values 
    median = df.logerror.median() 

    #calculating mean of property values 
    mean = df.logerror.mean()

    # difference between mean and median 
    difference = mean - median

    #provides data for table
    df = [["Median", median], 
        ["Mean", mean],
        ["Difference", difference]]
        
    #define header names
    col_names = ["Metric", "Value"]
  
    #display table
    print(tabulate(df, headers=col_names)) 
    

def two_variable_boxplots(df,column,Title):
    # box plot feature vs Property value 
    sns.boxplot(data=df, x=df[column], y=df['property_value']).set(title= Title)

def hists(df):
    '''this function will produce histograms of property values for each county'''
    sns.histplot(data=df, x="logerror")

def stats_property_location(train):
    '''this function will provide results of statistical test'''   

    # creating dataframe for each county's property values 
    la = train[train.fips == 'Los Angeles CA'].property_value
    oc = train[train.fips == 'Orange County CA'].property_value
    vc = train[train.fips == 'Ventura County CA'].property_value

    # results of statistical test 
    results, p = stats.kruskal(la, oc, vc)

    # print results of statistical test 
    print(f'Kruska Result = {results:.4f}')
    print(f'p = {p}')

def scatter_plot(train):
    '''this function will produce a scatter plot of data'''
    #visualization of sqft vs property value 
    sns.regplot(x="taxamount",
                y="logerror", 
                df=train).set(title='Taxamount and Value')

def correlation_stat_test(df,column):
    '''this function will produce results of pearsonr test'''

    #pearsonr test 
    corr, p = stats.pearsonr(df[column], df.logerror)
    print(f'Correlation Strength = {corr:.4f}')
    print(f'p = {p}')
 
