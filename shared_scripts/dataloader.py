# hydrological packages
import hydroeval as he
from hydrotools.nwm_client import utils 

# basic packages
import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq


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



def get_ML_Data(datapath, trainingfile):

    # Get Streamstats
    file = "Streamstats.csv"
    filepath = f"{datapath}/{file}"
    try:
        StreamStats = pd.read_csv(filepath)
    except:
        print("Data not found, retreiving from AWS S3")
        if not os.path.exists(datapath):
            os.makedirs(datapath, exist_ok=True)
        key = 'Streamstats/Streamstats.csv'      
        S3.meta.client.download_file(BUCKET_NAME, key,filepath)
        StreamStats = pd.read_csv(filepath)

    #Get processed training data 
    datapath = f"{HOME}/NWM_ML/Data/Processed"
    filepath = f"{datapath}/{trainingfile}"
    try:
        df = pd.read_parquet(filepath)
    except:
        print("Data not found, retreiving from AWS S3")
        if not os.path.exists(datapath):
            os.makedirs(datapath, exist_ok=True)
        key = "NWM_ML"+datapath.split("NWM_ML",1)[1]+'/'+trainingfile     
        print(key)  
        S3.meta.client.download_file(BUCKET_NAME, key,filepath)
        df = pd.read_parquet(filepath)

    try:
        df.pop('Unnamed: 0')
    except:
        print('df needs no processing')
    df['station_id'] = df['station_id'].astype('str')

    return df, StreamStats