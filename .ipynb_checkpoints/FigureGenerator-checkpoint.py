#script for making figures

# basic packages
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#load access key
HOME = os.path.expanduser('~')

#plot time series of regionally average obs and preds
def TS_plot(dictionary, model, plotname, title, freq, supply = False, fill_between = False):
    cfsday_AFday = 1.983
    cols = ['Obs_flow', 'NWM_flow', f"{model}_flow"]

    #Get keys from dictionary
    keys = list(dictionary.keys())
    #print(keys)

    #make figure
    fig, ax = plt.subplots(3,3, figsize=(10, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    
    ax = ax.ravel()
    
    if freq == 'D':
            units = 'cfs'
    else:
            units = 'Acre-Feet'

    for i in range(len(ax.ravel())):
        key = keys[i]
        RegionDF = dictionary[key].copy()
        RegionDF = RegionDF[cols]
        
        if freq != 'D':
            
        #Adjust for different time intervals here
            RegionDF = RegionDF*cfsday_AFday
            RegionDF = RegionDF.resample(freq).sum()

        if supply == True:
            RegionDF = RegionDF*cfsday_AFday
            units = 'Acre-Feet (x1000)'
            #set up cumulative monthly values
            RegionDF['Year'] = RegionDF.index.year
            RegionDF['Obs_flow'] = RegionDF['Obs_flow']/1000
            RegionDF['NWM_flow'] = RegionDF['NWM_flow']/1000
            

            #RegionDF = pd.DataFrame(columns=columns)

            for site in cols:
                RegionDF[site] = RegionDF.groupby(['Year'])[site].cumsum()        
        
        #fig.patch.set_facecolor('white')
        ax[i].plot(RegionDF.index.values, RegionDF['Obs_flow'].values, color = 'green')
        #ax[i].plot(RegionDF.index, RegionDF[f"{model}_flow"],  color = 'orange')
        ax[i].plot(RegionDF.index.values, RegionDF['NWM_flow'].values,  color = 'blue')
        ax[i].xaxis.set_major_locator(MonthLocator())
        ax[i].xaxis.set_major_formatter(DateFormatter('%m'))

        y1 =RegionDF['Obs_flow'].values
        y2 = RegionDF['NWM_flow'].values

        if fill_between == True:
            ax[i].fill_between(x=RegionDF.index.values, y1=y1, y2=y2, where=y1>y2, color='green',  interpolate=True, alpha=.75)
            ax[i].fill_between(x=RegionDF.index.values, y1=y1, y2=y2, where=y1<y2, color='blue',  interpolate=True, alpha=.75)

        
        if i == 0:
            ax[i].set_ylabel(f"Flow ({units})", fontsize = 12)
            ax[i].set_xticklabels([])
        
            
        if i == 3:
            ax[i].set_ylabel(f"Flow ({units})", fontsize = 12)
            #ax[i].set_xlabel('Date', fontsize = 12)
            #ax[i].tick_params(axis='x', rotation=45)
            
        if i < 6:
            ax[i].set_xticklabels([])
            
        if i == 6:
            ax[i].set_ylabel(f"Flow ({units})", fontsize = 12)
            ax[i].set_xlabel('Month', fontsize = 12)
            ax[i].tick_params(axis='x', rotation=45)
            ax[i].plot(RegionDF.index.values, RegionDF['Obs_flow'].values, color = 'green', label = 'Obs Flow ')
            #ax[i].plot(RegionDF.index.values, RegionDF[f"{model}_flow"].values,  color = 'orange', label = f"{model} flow" )
            ax[i].plot(RegionDF.index.values, RegionDF['NWM_flow'].values,  color = 'blue', label = f"NWM flow" )
            ax[i].legend( loc = 'lower center', bbox_to_anchor = (0, -0.0, 1, 0),  bbox_transform = plt.gcf().transFigure, ncol = 3)
            
        if i > 6:
            ax[i].set_xlabel('Time', fontsize = 12)
            ax[i].tick_params(axis='x', rotation=45)
          
            
        #ax[0,0].set_xlabel('Date', fontsize = 12)
        ax[i].set_title(f"NHD reach: {key}", fontsize = 14)
    fig.suptitle(title, fontsize = 16)
    figpath = f"{HOME}/NWM-ML/Predictions/Hindcast/{model}/Figures"
    if os.path.exists(figpath)==False:
        os.mkdir(figpath)
    plt.savefig(f"{figpath}/{plotname}_{freq}_{supply}.png", dpi = 600, bbox_inches = 'tight')
    plt.show()
    
    
#Parity plot of the model results
def Parity_plot(dictionary, model, plotname):

    #Get keys from dictionary
    keys = list(dictionary.keys())
    #print(keys)

    #make figure
    fig, ax = plt.subplots(3,3, figsize=(10, 10), gridspec_kw={'hspace': 0.4})
    
    ax = ax.ravel()

    for i in range(len(ax.ravel())):
        key = keys[i]
        RegionDF = dictionary[key]
        
        ymin = min(RegionDF['Obs_flow'])*0.9
        ymax = max(RegionDF['Obs_flow'])*1.1
        
        #fig.patch.set_facecolor('white')
        ax[i].scatter(RegionDF['Obs_flow'], RegionDF[f"{model}_flow"], alpha=0.35,  color = 'orange', s = 7)
        ax[i].scatter(RegionDF['Obs_flow'], RegionDF['NWM_flow'], alpha=0.35,  color = 'blue', s = 7)
        ax[i].plot((0,ymax),(0,ymax), linestyle = '--', color  = 'red')
        
        
        
        if i == 0:
            ax[i].set_ylabel('Prediction (cfs)', fontsize = 12)
            #ax[i].set_xticklabels([])
        
            
        if i == 3:
            ax[i].set_ylabel('Prediction (cfs)', fontsize = 12)

            
        #if i < 6:
            #ax[i].set_xticklabels([])
            
        if i == 6:
            ax[i].set_ylabel('Prediction (cfs)', fontsize = 12)
            ax[i].set_xlabel('Observed (cfs)', fontsize = 12)
            ax[i].tick_params(axis='x')
            ax[i].scatter(RegionDF['Obs_flow'], RegionDF[f"{model}_flow"], alpha=0.35, s = 7,  color = 'orange', label = f"{model} flow" )
            ax[i].scatter(RegionDF['Obs_flow'], RegionDF['NWM_flow'], alpha=0.35, s = 7,  color = 'blue', label = f"NWM flow" )
            ax[i].legend( loc = 'lower center', bbox_to_anchor = (0, -0.0, 1, 0),  bbox_transform = plt.gcf().transFigure, ncol = 2)
            
        if i > 6:
            ax[i].set_xlabel('Observed (cfs)', fontsize = 12)
            ax[i].tick_params(axis='x', rotation=45)
          
            
        #ax[0,0].set_xlabel('Date', fontsize = 12)
        ax[i].set_title(f"NHD reach {key}", fontsize = 14)

    plt.tight_layout()
    plt.savefig(f"./Predictions/Hindcast/{model}/Figures/{plotname}.png", dpi = 600, bbox_inches = 'tight')
    plt.show()