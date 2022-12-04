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
    * The data wasfrom the Codeup database using mySQL workbench
    * Each observation represents a single family home
    * Each column represents a feature of the property
    * I acquired this data on Wednesday November 30th 
 
* Prepare data
    * Removed nulls from rows then removed columns with most nulls
    * Outliers were not identified or intentionally removed
    * Year built column changed to age and imputed values to show age of property
    * Split data into train, validate and test(approx. 50/30/20)
    * Renamed multiple columns : bedrooms, bathrooms, complete square footage

* Explore data in search of drivers of propety value
   * Answer the following initial questions
       * What is the mean and median logerror?
       * Is there a correlation between taxamount and logerror?
       * Is there a signfiicant difference in mean logerror across the three counties?
       * Is there a significant difference in the mean logerror of homes in the location cluster number one compared to the population?
       * Is there a significant difference in the mean logerror of homes in foundation cluster number two compared to the population?
       
      
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
|LCluster #1|Cluster group one generated from features age, longitude, latitude|
|LCluster #2|Cluster group two generated from features sqft and calculatedlotsize|
|FCluster #4|Cluster group four generated from features age, longitude, latitude|



# Steps to Reproduce
1) Clone this repo
2) Acquire the data from mySQL workbench database 
3) Create env file with username, password and codeup host name 
4) Include the function below in your env file
def get_db_url(db, user = username, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
5) Put the data in the file containing the cloned repo
6) Run notebook

### Summary
- The median logerror is .0069
- Statistical evidence supports a correlation between taxamount and 
- The counties have signficant differences in their mean log errors
- Properties within location cluster 1 have a different mean log error to the population
- Properties within location cluster 2 have a different mean log error to the population


# Recommendations
* Compare data from 2017 from other years of zillow data and see if similar results are found
* Create models for each county


# Next Steps
* Create a model using the taxamounts column to predict logerror
* Spend more time researching why the logerror mean was highest in Orange County