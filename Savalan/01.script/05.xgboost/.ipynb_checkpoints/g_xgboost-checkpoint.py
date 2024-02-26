# This file created on 02/20/2024 by savalan

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import joblib
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

class XGBoostRegressorCV:
    def __init__(self, params, path=None):
        self.params = params
        self.model = xgb.XGBRegressor()
        self.best_model = None
        self.path = path

    def tune_hyperparameters(self, X, y, cv=3):
        """Performs GridSearchCV to find the best hyperparameters."""
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=self.params, 
                                   scoring='neg_mean_absolute_error',
                                   cv=cv, 
                                   n_jobs = -1,
                                   verbose=3)
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best RMSE: {grid_search.best_score_}")
        #save the model features
        joblib.dump(grid_search, self.path)

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
