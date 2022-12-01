from env import host, username, password 
import pandas as pd

def get_db_url(db, user = username, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def make_employees_db():
    return pd.read_sql('''SELECT *
FROM departments
LIMIT 10''', get_db_url('employees'))

def make_db(db, x):
    return pd.read_sql(
        x, get_db_url(db)
    )
