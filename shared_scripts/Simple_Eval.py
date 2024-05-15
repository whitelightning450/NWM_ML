#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
import hydroeval as he
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

def Model_Evaluation_Plots(DF, predictions):

    fontsize = 12

# Subplots.
    fig, ax = plt.subplots(1,2, 
                           figsize=(8, 4))
    plt.subplots_adjust(
                    wspace=0.4
                   )
    fig.patch.set_facecolor('white')

    #set min/max for y-axis of the predictions/observations
    ymin = min(DF['flow_cfs'])*1.1
    ymax = max(DF['flow_cfs'])*1.1
    
    #add color options
    colors = ['blue', 'orange', 'red','green']

    #hydrograph plot
    ax[0].plot(DF['Datetime'], DF['flow_cfs'],
                   c='green', alpha=0.35, label= 'Observed')

    # Add predictions to plot
    for pred in np.arange(0, len(predictions),1):
        ax[0].plot(DF['Datetime'], DF[predictions[pred]],
                   c=colors[pred], alpha=0.35, label=predictions[pred])

     # Add some parameters.
    ax[0].set_title('Streamflow Predictions', fontsize=fontsize)
    ax[0].set_xlabel('Date Time', fontsize=fontsize-2)
    ax[0].set_ylabel('Streamflow (cfs)', fontsize=fontsize-2,)
    ax[0].set_ylim(0, ymax)
    ax[0].xaxis.set_major_locator(MonthLocator())
    ax[0].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax[0].set_xticklabels(labels=pd.to_datetime(DF.Datetime).dt.to_period('M').unique(),rotation=90)
    ax[0].legend(fontsize=fontsize-2, loc='upper right')


    # Addscatter plot
    for pred in np.arange(0, len(predictions),1):
        ax[1].scatter(DF['flow_cfs'], DF[predictions[pred]],
                   c=colors[pred], alpha=0.35, label=predictions[pred])

     # Add some parameters.
    ax[1].set_title('Streamflow Predictions', fontsize=fontsize)
    ax[1].set_xlabel('Observations (cfs)', fontsize=fontsize-2)
    ax[1].set_ylabel('Predictions (cfs)', fontsize=fontsize-2,)
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_xlim(ymin, ymax)
    ax[1].legend(fontsize=fontsize-2, loc='upper right')
    
    #Add a 1:1 prediction:observation plot
    ax[1].plot((0,ymax),(0,ymax), linestyle = '--', color  = 'red')

    plt.show()
    
    
def Hydrograph_Evaluation_Plots(DF, predictions):

# Subplots.
    fig, ax = plt.subplots(1,1, figsize=(8, 7))
    fig.patch.set_facecolor('white')

    #set min/max for y-axis of the predictions/observations
    ymin = min(DF['flow_cfs'])*1.1
    ymax = max(DF['flow_cfs'])*1.1
    
    #add color options
    colors = ['blue', 'red','green']
    
    ax.plot(DF['DOY'], DF['flow_cfs'],
                   c='orange', alpha=0.35, label= 'Observed')

    # Add predictions to plot
    for pred in np.arange(0, len(predictions),1):
        ax.plot(DF['DOY'], DF[predictions[pred]],
                   c=colors[pred], alpha=0.35, label=predictions[pred])

     # Add some parameters.
    ax.set_title('Streamflow Predictions', fontsize=16)
    ax.set_xlabel('Time (DOY)', fontsize=14)
    ax.set_ylabel('Streamflow (cfs)', fontsize=14,)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=14, loc='upper right')

    plt.show()
    

#Define some key model performance metics: RMSE, PBias, MAE, MAPE
def RMSE(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        rmse = mean_squared_error(DF['flow_cfs'], DF[pred], squared=False)
        #print('RMSE for ', predictions[pred], ' is ', rmse, ' cfs')
        eval_dict[f"{pred}_rmse"] = rmse

    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def MAPE(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        mape = round(mean_absolute_percentage_error(DF['flow_cfs'], DF[pred])*100, 2)
        #print('Mean Absolute Percentage Error for ', predictions[pred], ' is ', mape, '%')
        eval_dict[f"{pred}_mape"] = mape
    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval
        
def PBias(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        pbias = he.evaluator(he.pbias,  DF[pred], DF['flow_cfs'])
        pbias = round(pbias[0],2)
        #print('Percentage Bias for ', predictions[pred], ' is ', pbias, '%')
        eval_dict[f"{pred}_pbias"] = pbias
    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def KGE(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        kge, r, alpha, beta = he.evaluator(he.kge,  DF[pred], DF['flow_cfs'])
        kge = round(kge[0],2)
        #print('Kling-Glutz Efficiency for ', predictions[pred], ' is ', kge)
        eval_dict[f"{pred}_kge"] = kge
    
    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def Key_Stats(DF, predictions):
    eval_dict = {}
    eval_dict['min_storage'] = min(DF['storage'])
    eval_dict['max_storage'] = max(DF['storage'])
    eval_dict['min_swe'] = min(DF['swe'])
    eval_dict['max_swe'] = max(DF['swe'])
    eval_dict['min_obs_flow'] = min(DF['flow_cfs'])
    eval_dict['max_obs_flow'] = max(DF['flow_cfs'])
    for pred in predictions:
        eval_dict[f"{pred}_min"] = min(DF[pred])
        eval_dict[f"{pred}_fmax"] = max(DF[pred])

    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def Simple_Eval(Preds_Dict, prediction_columns, modelname, supply = False):
    sites = list(Preds_Dict.keys())
    Eval_DF = pd.DataFrame()

    for site in sites:
        df = Preds_Dict[site].copy()

        if supply == True:
            #plot the predictions
            cols = ['Datetime', 'flow_cfs'] + prediction_columns
            df_CumSum = df[cols]
            cols.remove('Datetime')
            df_CumSum['Year'] = pd.to_datetime(df_CumSum['Datetime']).dt.year
            df_CumSum.set_index('Datetime', inplace = True)

            df_CumSum2 = pd.DataFrame(columns=cols)

            for col in cols:
                df_CumSum2[col] = df_CumSum.groupby(['Year'])[col].cumsum()
            df_CumSum2.reset_index(inplace=True)
            df.update(df_CumSum2)

        print(f"USGS site: {site}")
        Model_Evaluation_Plots(df, prediction_columns)

        #put the below into a DF so we can compare all sites..
        #Get RMSE from the model
        rmse = RMSE(df, prediction_columns)

        #Get Mean Absolute Percentage Error from the model
        mape = MAPE(df, prediction_columns)

        #Get Percent Bias from the model
        pbias = PBias(df, prediction_columns)

        #Get Kling-Gutz Efficiency from the model
        kge = KGE(df, prediction_columns)

        #Print key site characterstics
        stats = Key_Stats(df, prediction_columns)

        site_df = pd.DataFrame()

        evaldf = pd.concat([site_df, kge,rmse,mape,pbias,stats],axis = 1)
        evaldf['station_id'] = site

        Eval_DF = pd.concat([Eval_DF, evaldf])

    Eval_DF.sort_values(by = [f"{modelname}_flow_kge"], ascending=False, inplace = True)
    Eval_DF.set_index('station_id', inplace = True)
    return Eval_DF
        

