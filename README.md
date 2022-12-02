# Predicting logerror
 
# Project Description
My team is working to determine the drivers of the logerror of the zillow single family home prediction model. The goal is to identify the drivers of the logerror and create a model that more accurately predicts the logerror. 
 
# Project Goal
* Identify which features are drivers for predicting the logerror
* Develop and use machine learning model that accurately predicts the logerror.
* Deliver a report that explains what steps were taken and why there were taken.

 
# Initial Thoughts
Logerror will be dependent on the square feet of a home and the tax amount. 


# The Plan
* Acquire data from Codeup database using mySQL Workbench
 
* Prepare data
   * Changed a few column names for readability
       * beds
       * baths
       * sqft

* Explore data in search of drivers of propety value
   * Answer the following initial questions
       * What is the mean and median logerror?
       Is there a correlation between taxamount and logerror?
       * Is there a signfiicant difference in mean logerror across the three counties?
       * 
       * 
       * 
      
* Develop a Model to predict an accurate value of the property
   * Use drivers supported by statistical test results to build predictive models
   * Evaluate all models on train 
   * Evaluate top models on validate data 
   * Select the best model based on lowest Root Mean Squared Error
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Beds| Number of bedrooms in the home|
|Baths| Number of bathrooms in the home|
|sqft| The square footage of the home|
|Cluster #| |
|Cluster #| |
|Cluster #| |
|Cluster #| |


# Steps to Reproduce
1) Clone this repo
2) Acquire the data from mySQL workbench database 
3) Create env file with username, password and codeup host name 
4) Include the function below in your env file
def get_db_url(db, user = username, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
5) Put the data in the file containing the cloned repo
6) Run notebook