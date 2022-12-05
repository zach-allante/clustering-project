from env import host, username, password 
import pandas as pd

def get_db_url(db, user = username, host = host, password = password):
    '''This function will return the credentials to access the chosen database'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def make_employees_db():
    '''this function will read the first 10 lines of the data retrieved from SQL'''
    return pd.read_sql('''SELECT *
FROM departments
LIMIT 10''', get_db_url('employees'))

def make_db(db, x):
    '''this function will run the read sql code to retrieve a database'''
    return pd.read_sql(
        x, get_db_url(db)
    )
