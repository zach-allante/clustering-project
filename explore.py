#imports used for explore file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans




def bar_chart(df,column1,column2):
    '''This function is used to create a bar chart'''
    #title of chart
    plt.title("Fips and Logerror")
    #creating chart with data
    sns.barplot(x=column1, y=column2, data=df)
    #identify mean of column 2 
    mean_logerror = df[column2].mean()
    #style of chart
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
    '''this funcion will create a boxplot'''
    # box plot feature vs Property value 
    sns.boxplot(data=df, x=df[column], y=df['property_value']).set(title= Title)

def hists(df):
    '''this function will produce histograms of property values for each county'''
    sns.histplot(data=df, x="logerror")

def stats_property_location(train):
    '''this function will provide results of statistical test'''   

    # creating dataframe for each county's property values 
    la = train[train.fips == 6037].logerror
    oc = train[train.fips == 6059].logerror
    vc = train[train.fips == 6111].logerror

    # results of statistical test 
    results, p = stats.kruskal(la, oc, vc)

    # print results of statistical test 
    print(f'Kruska Result = {results:.4f}')
    print(f'p = {p}')

def scatter_plot(df):
    '''this function will produce a scatter plot of data'''
    #visualization of sqft vs property value 
    sns.scatterplot(x="logerror",
                y="taxamount",
                data =df).set(title='Taxamount and Value')

def correlation_stat_test(df,column):
    '''this function will produce results of pearsonr test'''

    #pearsonr test 
    corr, p = stats.pearsonr(df[column], df.logerror)
    print(f'Correlation Strength = {corr:.4f}')
    print(f'p = {p}')

def boxplots(df,theme,n_clust,chosen_clust):
    ''' This function will create a boxplot for visualization'''
    #creating clusters
    kmeans = KMeans(n_clusters= n_clust, random_state = 123)
    kmeans.fit(df[theme])
    df['cluster'] = kmeans.predict(df[theme])
    Cluster = df[df['cluster'] == chosen_clust]
    #creating plot
    sns.boxplot(Cluster.logerror)
    plt.show()
    sns.boxplot(df.logerror)
    plt.show()

def ttest(df,theme,n_clust,chosen_clust):
    ''' This function will return the results of a ttest run on the samples fed into it'''
    #defining alpha
    alpha = .05
    #creating clusters
    kmeans = KMeans(n_clusters= n_clust, random_state = 123)
    kmeans.fit(df[theme])
    df['cluster'] = kmeans.predict(df[theme])
    Clusterlog = df[df['cluster'] == chosen_clust].logerror
    overall_mean = df.logerror.mean()
    #statistical test 
    t,p = stats.ttest_1samp(Clusterlog,overall_mean)
    if p/2 > alpha:
        print("We fail to reject $H_{0}$")
    elif t < 0:
        print("We fail to reject $H_{0}$")
    else:
        print("We reject $H_{0}$")
    return(t, p)

