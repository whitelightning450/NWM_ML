# This file created on 01/14/2024 by savalan

# Import packages ==============================
# My Packages
from g_evaluation_metric import MAPE, RMSE, KGE, PBias

# Functions ==============================

def evtab(Eval_DF_mine, prediction_columns, nhdreach, observation_column, mod):

    #get annual supply diffs
    cfsday_AFday = 1.983
    
    #Get RMSE from the model
    rmse = RMSE(Eval_DF_mine, prediction_columns, observation_column)

    #Get Mean Absolute Percentage Error from the model
    mape = MAPE(Eval_DF_mine, prediction_columns, observation_column)

    #Get Percent Bias from the model
    pbias = PBias(Eval_DF_mine, prediction_columns, observation_column)

    #Get Kling-Gutz Efficiency from the model
    kge = KGE(Eval_DF_mine, prediction_columns, observation_column)
    
    #Get Volumetric values
    Eval_DF_mine.set_index('datetime', inplace = True, drop =True)
    flowcols = [f"{mod}_flow", 'flow_cfs', 'NWM_flow']
    SupplyEval = Eval_DF_mine[flowcols].copy()
    SupplyEval = SupplyEval*cfsday_AFday
    #set up cumulative monthly values
    SupplyEval['Year'] = SupplyEval.index.year

    for col_name in flowcols:
        SupplyEval[col_name] = SupplyEval.groupby(['Year'])[col_name].cumsum()  

    EOY_mod_vol_af = SupplyEval[f"{mod}_flow"].iloc[-1]
    EOY_obs_vol_af = SupplyEval["flow_cfs"].iloc[-1]
    EOY_nwm_vol_af = SupplyEval[f"NWM_flow"].iloc[-1]
    NWM_vol_diff_af = EOY_nwm_vol_af - EOY_obs_vol_af
    Mod_vol_diff_af = EOY_mod_vol_af - EOY_obs_vol_af
    NWM_Perc_diff = (NWM_vol_diff_af/EOY_obs_vol_af)*100
    Mod_Perc_diff = (Mod_vol_diff_af/EOY_obs_vol_af)*100
    
     #Get Performance Metrics from the model
    Srmse = RMSE(SupplyEval, prediction_columns, observation_column)
    Smape = MAPE(SupplyEval, prediction_columns, observation_column)
    Spbias = PBias(SupplyEval, prediction_columns, observation_column)
    Skge = KGE(SupplyEval, prediction_columns, observation_column)
    
    
    #save model performance
    sitestats = [Eval_DF_mine.iloc[0, 1], nhdreach, rmse[0], rmse[1],  pbias[0], pbias[1], kge[0], kge[1], mape[0],mape[1]]

    
    Supplystats = [Eval_DF_mine.iloc[0, 1], nhdreach, Srmse[0], Srmse[1],  Spbias[0], Spbias[1], Skge[0], Skge[1], Smape[0],  
                 Smape[1],EOY_obs_vol_af, EOY_nwm_vol_af,EOY_mod_vol_af,NWM_vol_diff_af,Mod_vol_diff_af, NWM_Perc_diff, Mod_Perc_diff ]


    return sitestats, Supplystats