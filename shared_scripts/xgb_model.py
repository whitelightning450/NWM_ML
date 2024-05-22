# This file created on 02/20/2024 by savalan

import numpy as np
from hydrotools.nwm_client import utils 
import xgboost as xgb
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pickle as pkl
from xgboost import XGBRegressor
from scipy import optimize
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#Shared/Utility scripts
import os
import boto3
import s3fs
import sys
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

class XGBoostRegressorCV:
    def __init__(self, params, path=None):
        self.params = params
        self.model = xgb.XGBRegressor()
        #self.model = xgb.XGBRegressor(tree_method="hist", device="cuda"
        self.best_model = None
        self.path = path

    def tune_hyperparameters(self, X, y, cv=3):
        """Performs GridSearchCV to find the best hyperparameters."""
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=self.params, 
                                   scoring='neg_mean_absolute_error', #change to mean squared error
                                   cv=cv, 
                                   n_jobs = -1,
                                   verbose=3)
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best RMSE: {grid_search.best_score_}")
        #save the model features
        #joblib.dump(grid_search, self.path)
        pkl.dump(grid_search, open(self.path, "wb")) 

    def train(self, X, y, parameters={}):
        """Trains the model using the best hyperparameters found."""
        if self.best_model:
            self.best_model.fit(X, y)
        else:
            eta = parameters.best_params_['eta']
            max_depth =parameters.best_params_['max_depth']
            n_estimators = parameters.best_params_['n_estimators']
            self.model.fit(X, y, n_estimators=n_estimators, max_depth=max_depth, eta=eta, verbose=True)
            print("Please tune hyperparameters first.")

    def predict(self, X):
        """Predicts using the trained XGBoost model on the provided data."""
        if self.best_model:
            return self.best_model.predict(X)
        else:
            print("Model is not trained yet. Please train the model first.")
            return None

    def evaluate(self, X, y):
        """Evaluates the trained model on a separate test set."""
        if self.best_model:
            
            # define model evaluation method
            cv = RepeatedKFold(n_splits=10, n_repeats=3)
            
            # evaluate model
            scores = cross_val_score(self.best_model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

            # Caculate model performance and force scores to be positive
            print('Mean MAE: %.3f (%.3f)' % (abs(scores.mean()), scores.std()) )

        else:
            print("Model is not trained yet. Please train the model first.")


def XGB_Train(model_path, station_index_list, x_train, y_train, tries, hyperparameters, perc_data):
    start_time = time.time()

    # Start running the model several times. 
    for try_number in range(tries):
        print(f'Trial Number {try_number} ==========================================================')
        
        # # Set the optimizer, create the model, and train it. 
        xgboost_model = XGBoostRegressorCV(hyperparameters, f"{model_path}/best_model_hyperparameters.pkl")
        new_data_len = int(len(x_train) * perc_data) #determine hyperprams using 25% of the data
        print(f"Tuning hyperparametetrs on {perc_data*100}% of training data")
        xgboost_model.tune_hyperparameters(x_train.iloc[:new_data_len], y_train.iloc[:new_data_len])
        xgboost_model.evaluate(x_train.iloc[:new_data_len], y_train.iloc[:new_data_len])
        print('Training model with optimized hyperparameters')
        xgboost_model.train(x_train, y_train)
        print('Saving Model')
        
        #adjust this to match changing models
        pkl.dump(xgboost_model, open(f"{model_path}/best_model.pkl", "wb"))  

    print('Run is Done!' + "Run Time:" + " %s seconds " % (time.time() - start_time))


def XGB_Predict(model_path,modelname, df, x_test, y_test, test_years, StreamStats, station_index_list):

    #Load model
    xgboost_model = pkl.load(open(f"{model_path}/best_model.pkl", "rb"))
    df_test = df[df.datetime.dt.year.isin(test_years)]
    Preds_Dict = {}
    for station_index, station_number in enumerate(station_index_list.drop_duplicates()):
        index = station_index_list == station_number # Finind the rows that have this station number.
        predictions = xgboost_model.predict(x_test[index])
        
        # Invert scaling for actual and concat it with the rest of the dataset. 
        predictions = pd.DataFrame(predictions, columns=[f"{modelname}_flow"])

        predictions[f"{modelname}_flow"][predictions[f"{modelname}_flow"]< 0] =0.1

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
        df_test['nwm_feature_id'] = nhdreach
        Dfs = [predictions.reset_index(drop=True),df_test[df_test['station_id']==station_number].reset_index(drop=True)]
        Preds_Dict[station_number] = pd.concat(Dfs, axis=1)

        #reorganize columns
        Preds_Dict[station_number].pop('datetime')
        Preds_Dict[station_number].insert(1, f"{modelname}_flow", Preds_Dict[station_number].pop(f"{modelname}_flow"))
        Preds_Dict[station_number].insert(1, "NWM_flow", Preds_Dict[station_number].pop("NWM_flow"))
        Preds_Dict[station_number].insert(1, "flow_cfs", Preds_Dict[station_number].pop("flow_cfs"))
        Preds_Dict[station_number].insert(1, "nwm_feature_id", Preds_Dict[station_number].pop("nwm_feature_id"))
        Preds_Dict[station_number].insert(1, "station_id", Preds_Dict[station_number].pop("station_id"))  
    
    #save predictions as compressed pkl file
    pred_path = f"{HOME}/NWM_ML/Predictions/Hindcast/{modelname}/Multilocation"
    file_path = f"{pred_path}/{modelname}_predictions.pkl"
    if os.path.exists(pred_path) == False:
        os.makedirs(pred_path)
    with open(file_path, 'wb') as handle:
        pkl.dump(Preds_Dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return Preds_Dict
  