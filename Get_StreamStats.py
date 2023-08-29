#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataretrieval.nwis as nwis
##https://streamstats-python.readthedocs.io/en/latest/gallery_vignettes/plot_get_characteristics.html
import streamstats
import pandas as pd    
import numpy as np
import time
from progressbar import ProgressBar


def get_USGS_site_info(site_ids, Preloaded_data):

    #set up Pandas DF for state streamstats

    Streamstats_cols = ['NWIS_siteid','Lat', 'Long', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest', 'Perc_Develop',
                     'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']

    NWIS_Stats = pd.DataFrame(columns = Streamstats_cols)


    print('Calculating NWIS streamflow id characteristics for ', len(site_ids), 'sites')
    
    if Preloaded_data == False:

        pbar = ProgressBar()
        for site in pbar(site_ids):
            print('NWIS site: ', site)
            #try:
            NWISinfo = nwis.get_record(sites=site, service='site')

            lat, lon = NWISinfo['dec_lat_va'][0],NWISinfo['dec_long_va'][0]

            #This sources the prestored data
            try:
                ws = streamstats.Watershed(lat=lat, lon=lon)
            except:
                print('502 error, StreamStats down, using backup files')
                ws = pd.read_csv('./Data/StreamStats_All.csv')
                ws['NWIS_site_id'] = ws['NWIS_site_id'].astype(str)
                ws = ws[ws.NWIS_site_id == site]

            NWISindex = ['NWIS_site_id', 'Lat', 'Long', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest', 'Perc_Develop',
                         'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']

            print('Retrieving Drainage Area')
            try:
                darea = ws.get_characteristic('DRNAREA')['value']
            except KeyError:
                darea = np.nan
            except ValueError:
                darea = np.nan
            except AttributeError:
                darea = ws['Drainage_area_mi2'].values[0]

            print('Retrieving Mean Catchment Elevation')
            try:
                elev = ws.get_characteristic('ELEV')['value']
            except KeyError:
                elev = np.nan
            except ValueError:
                elev = np.nan
            except AttributeError:
                elev = ws['Mean_Basin_Elev_ft'].values[0]

            print('Retrieving Catchment Land Cover Information')
            try:
                forest = ws.get_characteristic('FOREST')['value']
            except KeyError:
                forest = np.nan
            except ValueError:
                forest = np.nan
            except AttributeError:
                forest = ws['Perc_Forest'].values[0]

            try:
                dev_area = ws.get_characteristic('LC11DEV')['value']
            except KeyError:
                dev_area = np.nan
            except ValueError:
                dev_area = np.nan
            except AttributeError:
                dev_area = ws['Perc_Develop'].values[0]

            try:
                imp_area = ws.get_characteristic('LC11IMP')['value']
            except KeyError:
                imp_area = np.nan
            except ValueError:
                imp_area = np.nan
            except AttributeError:
                imp_area = ws['Perc_Imperv'].values[0]

            try:
                herb_area = ws.get_characteristic('LU92HRBN')['value']
            except KeyError:
                herb_area = np.nan
            except ValueError:
                herb_area = np.nan
            except AttributeError:
                herb_area = ws['Perc_Herbace'].values[0]

            print('Retrieving Catchment Topographic Complexity')
            try:
                perc_slope = ws.get_characteristic('SLOP30_10M')['value']
            except KeyError:
                perc_slope = np.nan
            except ValueError:
                perc_slope = np.nan
            except AttributeError:
                perc_slope = ws['Perc_Slop_30'].values[0]

            print('Retrieving Catchment Average Precip')
            try:
                precip = ws.get_characteristic('PRECIP')['value']
            except KeyError:
                precip = np.nan
            except ValueError:
                precip = np.nan
            except AttributeError:
                precip = ws['Mean_Ann_Precip_in'].values[0]


            NWISvalues = [site,
                          lat,
                          lon,
                          darea, 
                          elev,forest, 
                          dev_area,
                          imp_area, 
                          herb_area,
                          perc_slope,
                          precip]

            print(NWISvalues)
            Catchment_Stats = pd.DataFrame(data = NWISvalues, index = NWISindex).T

            NWIS_Stats = NWIS_Stats.append(Catchment_Stats)
        

    if Preloaded_data == True:
        print('StreamStats downloaded files')
        ws = pd.read_csv('./Data/StreamStats_All.csv')
        ws['NWIS_site_id'] = ws['NWIS_site_id'].astype(str)
        pbar = ProgressBar()
        for site in pbar(site_ids):
            print('NWIS site: ', site)
            NWISinfo = nwis.get_record(sites=site, service='site')

            lat, lon = NWISinfo['dec_lat_va'][0],NWISinfo['dec_long_va'][0]
            ws1 = ws[ws.NWIS_site_id == site].copy()

            NWISindex = ['NWIS_site_id', 'Lat', 'Long', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest', 'Perc_Develop',
                         'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']

            darea = ws1['Drainage_area_mi2'].values[0]
            elev = ws1['Mean_Basin_Elev_ft'].values[0]
            forest = ws1['Perc_Forest'].values[0]
            dev_area = ws1['Perc_Develop'].values[0]
            imp_area = ws1['Perc_Imperv'].values[0]
            herb_area = ws1['Perc_Herbace'].values[0]
            perc_slope = ws1['Perc_Slop_30'].values[0]
            precip = ws1['Mean_Ann_Precip_in'].values[0]


            NWISvalues = [site,
                          lat,
                          lon,
                          darea, 
                          elev,forest, 
                          dev_area,
                          imp_area, 
                          herb_area,
                          perc_slope,
                          precip]

            print(NWISvalues)
            Catchment_Stats = pd.DataFrame(data = NWISvalues, index = NWISindex).T

            NWIS_Stats = NWIS_Stats.append(Catchment_Stats)


    colorder =['NWIS_site_id', 'Lat', 'Long', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest', 
               'Perc_Develop','Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']



    NWIS_Stats = NWIS_Stats[colorder]

    NWIS_Stats.reset_index(drop = True, inplace = True)
    
    return NWIS_Stats


# In[ ]:




