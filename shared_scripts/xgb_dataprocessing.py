# hydrological packages
import hydroeval as he
from hydrotools.nwm_client import utils 

# basic packages
import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import bz2file as bz2

# system packages
from progressbar import ProgressBar
from datetime import datetime, date
import datetime
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import platform
import time

# data analysi packages
from scipy import optimize
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



#Shared/Utility scripts
import sys
sys.path.insert(0, '../..') #sys allows for the .ipynb file to connect to the shared folder files
from shared_scripts import Simple_Eval

#load access key
HOME = os.path.expanduser('~')


def xgb_train_test(df, test_years, target, input_columns):

    #Select training data
    x_train = df[~df.datetime.dt.year.isin(test_years)]
    x_train.pop('station_id')
    x_train.pop('datetime')
    #shuffle data
    x_train = x_train.sample(frac = 1, random_state = 69)
    y_train = x_train[target]
    x_train.pop(target)
    x_train = x_train[input_columns]

    #select testing data
    x_test = df[df.datetime.dt.year.isin(test_years)]
    station_index_list = x_test['station_id']
    x_test.pop('station_id')
    x_test.pop('datetime')
    y_test = x_test[target]
    x_test.pop(target)
    x_test = x_test[input_columns]
    


    return x_train, y_train, x_test, y_test, station_index_list

