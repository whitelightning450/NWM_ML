# basic packages
import numpy as np
import pandas as pd
import os
import time
from hydrotools.nwm_client import utils 


# system packages
from datetime import datetime, date
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")


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
import boto3
import s3fs

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



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def model_arch(layers):
    input_shape, LD1, LD2, LD3, LD4, LD5, LD6 = layers
    # Build the model
    model = nn.Sequential(
        nn.Linear(input_shape, LD1),
        nn.ReLU(),
        nn.Linear(LD1, LD2),
        nn.ReLU(),
        nn.Linear(LD2, LD3),
        nn.ReLU(),
        nn.Linear(LD3, LD4),
        nn.ReLU(),
        nn.Linear(LD4, LD5),
        nn.ReLU(),
        nn.Linear(LD5, LD6),
        nn.ReLU(),
        nn.Linear(LD6, 1)
    ).to(DEVICE)

    return model


def mlp_train(x_train_scaled_t,y_train_scaled_t, layers, params, loss_func, model_path, modelname):
    start_time = time.time()
    learning_rate, decay, epochs, batch_size = params

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(x_train_scaled_t, y_train_scaled_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )

    #load model
    model = model_arch(layers)

    # Define loss and optimizer
    criterion = loss_func
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    #save model
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    torch.save(model.state_dict(), f"{model_path}/{modelname}_model.pkl")

    print('finish')
    print("Run Time:" + " %s seconds " % (time.time() - start_time))



def mlp_predict(test_years, layers, model_path, modelname, stations, x_test_temp, x_test_scaled, y_test_temp_1,StreamStats, station_index_list):

    #load model
    model = model_arch(layers)

    model.load_state_dict(torch.load(f"{model_path}/{modelname}_model.pkl"))



    Preds_Dict = {}
    for station_number in station_index_list.drop_duplicates():
        index = station_index_list == station_number
        X_test = x_test_temp[index]
        X_test_scaled_t = torch.Tensor(x_test_scaled[index])
        X_test_scaled_t = X_test_scaled_t.to(DEVICE)
        l = len(y_test_temp_1.values)
        y_test = torch.Tensor(np.array(y_test_temp_1.values).reshape(l,1))
        y_test = y_test.to(DEVICE)

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions_scaled= model(X_test_scaled_t)

        # Invert scaling for actual
        #load scalers and scale
        scalername_y = "scaler_y.save"
        scalerfilepath_y = f"{model_path}/{scalername_y}"

        #load scalers
        scaler_y = joblib.load(scalerfilepath_y)

        #scale the testing data
        predictions = scaler_y.inverse_transform(predictions_scaled.to('cpu').numpy())
        predictions[predictions<0] = 0

        #print('Model Predictions complete')

        predictions = pd.DataFrame(predictions, columns=[f"{modelname}_flow"])

        #save predictions, need to convert to NHDPlus reach - Need to add Datetime column and flow predictions
        #make daterange
        dates = pd.date_range(pd.to_datetime(f"{test_years[0]}-01-01"), periods=len(predictions)).strftime("%Y-%m-%d").tolist()
        predictions['Datetime'] = dates
            
        #get reach id for model eval
        nhdreach = utils.crosswalk(usgs_site_codes=station_number)
        nhdreach = nhdreach['nwm_feature_id'].iloc[0]

        #put columns in correct order
        cols = ['Datetime', f"{modelname}_flow"]
        predictions = predictions[cols]

        #save predictions to AWS so we can use CSES
        state = StreamStats['state_id'][StreamStats['NWIS_site_id'].astype(str)== station_number].values[0].lower()
        csv_key = f"{modelname}/NHD_segments_{state}.h5/{modelname[:3]}_{nhdreach}.csv"
        predictions.to_csv(f"s3://{BUCKET_NAME}/{csv_key}", index = False,  storage_options={'key': ACCESS['Access key ID'][0],
                                'secret': ACCESS['Secret access key'][0]})

        #Concat DFS and put into dictionary
        x_test_temp['nwm_feature_id'] = nhdreach
        Dfs = [predictions.reset_index(drop=True),x_test_temp[x_test_temp['station_id']==station_number].reset_index(drop=True)]
        Preds_Dict[station_number] = pd.concat(Dfs, axis=1)

        #reorganize columns
        Preds_Dict[station_number].pop('datetime')
        Preds_Dict[station_number].insert(1, f"{modelname}_flow", Preds_Dict[station_number].pop(f"{modelname}_flow"))
        Preds_Dict[station_number].insert(1, "NWM_flow", Preds_Dict[station_number].pop("NWM_flow"))
        Preds_Dict[station_number].insert(1, "flow_cfs", Preds_Dict[station_number].pop("flow_cfs"))
        Preds_Dict[station_number].insert(1, "nwm_feature_id", Preds_Dict[station_number].pop("nwm_feature_id"))
        Preds_Dict[station_number].insert(1, "station_id", Preds_Dict[station_number].pop("station_id"))

    #push data to AWS so we can use CSES
    
    
    #save predictions as compressed pkl file
    pred_path = f"{HOME}/NWM_ML/Predictions/Hindcast/{modelname}/Multilocation"
    file_path = f"{pred_path}/{modelname}_predictions.pkl"
    if os.path.exists(pred_path) == False:
        os.makedirs(pred_path)
    with open(file_path, 'wb') as handle:
        pkl.dump(Preds_Dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return Preds_Dict