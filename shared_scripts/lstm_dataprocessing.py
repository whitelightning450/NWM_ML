#basic packagers
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from time import process_time 
import joblib

#modeling packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#packages to load AWS data
import boto3
import os
from botocore import UNSIGNED 
from botocore.client import Config
import os
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

#Set Global Variables
HOME = os.path.expanduser('~')
KEYPATH = "NWM_ML/AWSaccessKeys.csv"
ACCESS_KEY = pd.read_csv(f"{HOME}/{KEYPATH}")

#AWS Data Connectivity
#start session
SESSION = boto3.Session(
    aws_access_key_id=ACCESS_KEY['Access key ID'][0],
    aws_secret_access_key=ACCESS_KEY['Secret access key'][0]
)
s3 = SESSION.resource('s3')

BUCKET_NAME = 'streamflow-app-data'
BUCKET = s3.Bucket(BUCKET_NAME) 
S3 = boto3.resource('s3')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#read parquet file
def readdata(filepath, drop = False):
    obj = BUCKET.Object(filepath)
    body = obj.get()['Body']
    df = pd.read_csv(body)

    if drop == True:
        df.pop('Unnamed: 0')

    return df

#Combine DFs
def df_combine(USGS, NWM):
    #changes date as str to datetime object
    #USGS
    USGS['Datetime'] = USGS['Datetime'].astype('datetime64[ns]')
    #set index to datetime
    USGS.set_index('Datetime', inplace = True)
    #select streamflow
    cols =['USGS_flow']
    USGS = USGS[cols]
    #remove NaN values
    USGS.dropna(inplace = True)

    #NWM
    NWM['Datetime'] = NWM['Datetime'].astype('datetime64[ns]')
    #set index to datetime
    NWM.set_index('Datetime', inplace = True)
    #select streamflow
    cols =['NWM_flow']
    NWM = NWM[cols]
    #remove NaN values
    NWM.dropna(inplace = True)

    #combine NWM and USGS DF by datetime
    df = pd.concat([USGS, NWM], axis =1)
    df.dropna(inplace = True)

    return df

#create tensors/lookback out of training data for pytorch
def create_lookback_univariate(dataset, lookback):
    '''
    Transform a time series into a prediction dataset
    Args:
        dataset - a numpy array of time series, first dimension is the time step
        lookback -  size of window for prediction
    '''
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature, target = dataset[i:i+lookback], dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
        
    return torch.tensor(X).to(DEVICE), torch.tensor(y).to(DEVICE)

def create_lookback_multivariate(dataset, lookback):
    X, y = [],[]
    for i in range(len(dataset)-lookback):
        # find the end of this pattern
        end_ix = i + lookback
        if end_ix > len(dataset):
            break
        features, targets = dataset[i:i+lookback, :-1], dataset[i+lookback, -1]
        X.append(features)
        y.append(targets)
    return np.array(X), np.array(y)

# split a multivariate sequences into train/test
def Multisite_DataProcessing(df, input_columns, target, lookback, test_years, model_path, scalertype):

    #convert dates to datetime format
    df.datetime = pd.to_datetime(df.datetime)

    # #reset index to clean up df
    df.reset_index( inplace =  True, drop = True)

    scalername_x = "scaler_x.save"
    scalername_y = "scaler_y.save"
    x_scaler_path = f"{model_path}/{scalername_x}"
    y_scaler_path = f"{model_path}/{scalername_y}"

    #create training df to create scalers
    train_df = df[~df.datetime.dt.year.isin(test_years)].copy()
    train_features_df = train_df[input_columns].copy()
    train_targets_df = train_df[target].copy()


    if scalertype == 'MinMaxScaler':
        #scale X training data and save
        xscaler = MinMaxScaler()
        x_scaler = xscaler.fit(train_features_df)
        joblib.dump(x_scaler, x_scaler_path)

        #scale Y training data and save
        yscaler = MinMaxScaler()
        y_scaler = yscaler.fit(train_targets_df)
        joblib.dump(y_scaler, y_scaler_path)
        print(f"{scalertype} applied")


    if scalertype == 'StandardScaler':
        #scale X training data and save
        xscaler = StandardScaler()
        x_scaler = xscaler.fit(train_features_df)
        joblib.dump(x_scaler, x_scaler_path)

        #scale Y training data and save
        yscaler = StandardScaler()
        y_scaler = yscaler.fit(train_targets_df)
        joblib.dump(y_scaler, y_scaler_path)
        print(f"{scalertype} applied")


    #separate each site to create the appropariate lookback, scale as necessary
    sites = list(df['station_id'].unique())
    sites_feat_train_dict = {}
    sites_feat_test_dict = {}
    sites_targ_train_dict = {}
    sites_targ_test_dict = {}
    for site in sites:
        #select site and separate into features/target to scale
        site_df = df[df['station_id'] == site]

        #Count the len of dataset for x years to determine train/test split without reducing lookback data
        test_len = len(site_df[site_df.datetime.dt.year.isin(test_years)])
        
        #Separate features/targets
        features_df = site_df[input_columns]
        target_df = site_df[target]
        #scale the features/target
        features_scaled = x_scaler.transform(features_df)
        target_scaled = y_scaler.transform(target_df)

        #stack into numpy for lookback
        scaled_array = np.hstack((features_scaled, target_scaled))

        #split features/target with appropriate lookback
        X, y = create_lookback_multivariate(scaled_array, lookback)

        #add sites to dic for further processing
        sites_feat_train_dict[site] = X[:-test_len,:,:]
        sites_feat_test_dict[site] = X[-test_len:,:,:]
        sites_targ_train_dict[site] = y[:-test_len]
        sites_targ_test_dict[site] = y[-test_len:]

    #combine sites into one dataframe for model training
    X_train =np.zeros((1,lookback,len(input_columns)))
    y_train = np.zeros((1,))
    for site in sites:
        X_train = np.concatenate([X_train, sites_feat_train_dict[site]], axis = 0)
        y_train = np.concatenate([y_train, sites_targ_train_dict[site]], axis = 0)

    #remove filler zero
    X_train = X_train[1:,:,:]
    y_train = y_train[1:,]

    # feats = X_train.shape[2]
    # for f in np.arange(0,feats,1):
    #     print(f)
    #     print(f"Features Max: {max(X_train[:,0,f])}, Min: {min(X_train[:,0,f])}")
    # print(f"Target Max: {max(y_train[:,])}, Min: {min(y_train[:,])}")


    # #do a random shuffle - no need for this, you can do it in the tensorflow
    # if random_shuffle == True:
    #     np.random.seed(69)
    #     train = list(zip(X_train, y_train))
    #     np.random.shuffle(train)

    #     X_train, y_train = zip(*train)
    #     X_train = np.array(X_train)
    #     y_train = np.array(y_train)

    #need to convert to float32, tensors of the expected shape, and make sure they are on the device
    X_train = Variable(torch.from_numpy(X_train).float(), requires_grad=False).to(DEVICE)
    y_train = Variable(torch.from_numpy(y_train).float(), requires_grad=False).to(DEVICE)
    
    return X_train, sites_feat_test_dict, y_train, sites_targ_test_dict