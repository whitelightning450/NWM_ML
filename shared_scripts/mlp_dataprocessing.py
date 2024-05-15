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
import joblib

# deep learning packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Shared/Utility scripts
import sys
import boto3
import s3fs
sys.path.insert(0, '../..') #sys allows for the .ipynb file to connect to the shared folder files
from shared_scripts import Simple_Eval

#load access key
HOME = os.path.expanduser('~')
KEYPATH = "NWM_ML/AWSaccessKeys.csv"
ACCESS = pd.read_csv(f"{HOME}/{KEYPATH}")

#start session
SESSION = boto3.Session(
    aws_access_key_id=ACCESS['Access key ID'][0],
    aws_secret_access_key=ACCESS['Secret access key'][0],
)
S3 = SESSION.resource('s3')
#AWS BUCKET information
BUCKET_NAME = 'streamflow-app-data'
BUCKET = S3.Bucket(BUCKET_NAME)

#s3fs
fs = s3fs.S3FileSystem(anon=False, key=ACCESS['Access key ID'][0], secret=ACCESS['Secret access key'][0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def mlp_scaler(df, test_years, target, input_columns, model_path, scalertype = 'MinMax'):

    #Select training data
    x_train_temp = df[~df.datetime.dt.year.isin(test_years)]
    x_train_temp.pop('station_id')
    x_train_temp.pop('datetime')
    y_train_temp = x_train_temp[target]
    x_train_temp.pop(target)
    x_train_temp = x_train_temp[input_columns]

    #Convert dataframe to numpy, scale, save scalers
    y_train = y_train_temp.to_numpy()
    x_train = x_train_temp.to_numpy()

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    scalerfilepath_x = f"{model_path}/scaler_x.save"
    scalerfilepath_y = f"{model_path}/scaler_y.save"

    if scalertype == 'MinMax':
        scaler = MinMaxScaler() #potentially change scalling...StandardScaler
        x_train_scaled = scaler.fit_transform(x_train)
        joblib.dump(scaler, scalerfilepath_x)

        scaler = MinMaxScaler() #potentially change scalling...StandardScaler
        y_scaled_train = scaler.fit_transform(y_train.reshape(-1, 1))
        joblib.dump(scaler, scalerfilepath_y)  

    if scalertype == 'Standard':
        scaler = StandardScaler() #potentially change scalling...StandardScaler
        x_train_scaled = scaler.fit_transform(x_train)
        joblib.dump(scaler, scalerfilepath_x)

        scaler = StandardScaler() #potentially change scalling...StandardScaler
        y_scaled_train = scaler.fit_transform(y_train.reshape(-1, 1))
        joblib.dump(scaler, scalerfilepath_y) 

    print(y_scaled_train.shape)
    print(x_train_scaled.shape)

    return x_train_scaled, y_scaled_train

def mlp_testscaler(df, test_years, target, input_columns, model_path):
    #Get water year for testing from larger dataset
    x_test_temp = df[df.datetime.dt.year.isin(test_years)]
    x_test_temp_1 = x_test_temp.copy()
    station_index_list = x_test_temp_1['station_id']
    x_test_temp_1.pop('station_id')
    x_test_temp_1.pop('datetime')

    #Get target variable (y) and convert to numpy arrays
    y_test_temp_1 = x_test_temp_1[target]
    x_test_temp_1.pop(target)
    x_test_temp_1 = x_test_temp_1[input_columns]
    x_test_1_np = x_test_temp_1.reset_index(drop=True).to_numpy()
    #y_test_1_np = y_test_temp_1.reset_index(drop=True).to_numpy()


    #load scalers and scale
    scalerfilepath_x = f"{model_path}/scaler_x.save"

    #load scalers
    scaler_x = joblib.load(scalerfilepath_x)
    # scaler_y = joblib.load(scalerfilepath_y)

    #scale the testing data
    x_test_1_scaled = scaler_x.fit_transform(x_test_1_np)
    #y_scaled_test_1 = scaler_y.fit_transform(y_test_1_np.reshape(-1, 1))
    #print(y_scaled_test_1.shape)
    print(x_test_1_scaled.shape)

    return x_test_1_scaled, y_test_temp_1, x_test_temp, station_index_list