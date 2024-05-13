# push predictions to AWS
import pandas as pd
import boto3
import numpy as np
import os
import warnings
from pathlib import Path
from progressbar import ProgressBar
warnings.filterwarnings("ignore")

def Predictions2AWS(model):

    #load access key
    home = os.path.expanduser('~')
    keypath = "SageMaker/AWS/AWSaccessKeys.csv"
    access = pd.read_csv(f"{home}/{keypath}")

    #start session
    session = boto3.Session(
        aws_access_key_id=access['Access key ID'][0],
        aws_secret_access_key=access['Secret access key'][0],
    )
    s3 = session.resource('s3')
    #AWS bucket information
    bucket_name = 'streamflow-app-data'
    bucket = s3.Bucket(bucket_name)

    #push NSM data to AWS

    AWSpath = f"{model}/"
    path = f"./Predictions/Hindcast/{AWSpath}"
    files = []
    folders =[]
    files = []
    for file in os.listdir(path):
        if file.endswith("csv"):
            files.append(file)



    #Load and push to AWS
    #Load and push to AWS
    pbar = ProgressBar()
    print('Pushing files to AWS')
    for file in pbar(files):
        filepath = f"{path}{file}"
        s3.meta.client.upload_file(Filename= filepath, Bucket=bucket_name, Key=f"{AWSpath}{file}")