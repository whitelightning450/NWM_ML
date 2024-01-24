
import dataretrieval.nwis as nwis
import streamstats
import pandas as pd
from progressbar import ProgressBar
from time import sleep
from tqdm import tqdm
import numpy as np


def get_USGS_site_info(site_ids):
    
    col_num = 11
    nwis_coordinates = {}
    nwis_value_all = np.full([len(site_ids), col_num], np.nan)
    error_value = 1
    nwis_index = ['station_id', 'Lat', 'Long', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest',
                 'Perc_Develop', 'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']
    variable_list = ['DRNAREA', 'ELEV', 'FOREST', 'LC11DEV', 'LC11IMP', 'LU92HRBN', 'SLOP30_10M', 'PRECIP']
    print('Calculating NWIS streamflow id characteristics for ', len(site_ids), 'sites')
    
    for site_index, site in enumerate(tqdm(site_ids)):
        nwis_value_all[site_index, 0] = site
        nwis_info = nwis.get_record(sites=site, service='site')
        nwis_value_all[site_index, 1], nwis_value_all[site_index, 2] = nwis_info['dec_lat_va'][0], nwis_info['dec_long_va'][0]
        try:
            ws = streamstats.Watershed(lat=nwis_value_all[site_index, 1], lon=nwis_value_all[site_index, 2])
        except:
            print('502 error, StreamStats down, using backup files')

        for col_index in range(3, col_num): 
            
            error_value = 1
            while 0 < error_value <= 5:

                try:
                    nwis_value_all[site_index, col_index] = ws.get_characteristic(variable_list[col_index - 3])['value']
                    error_value = 0

                except Exception as error_name:
                    print('Try Again: The error is with ' + variable_list[col_index - 3] +  'and error is ' + str(error_name))
                    error_value += 1
                    pass

                sleep(2)  # wait for 2 seconds before trying to fetch the data again
                
        error_value = 1
        nwis_coordinates[site] = np.nan
        while 0 < error_value <= 5:
            try:
                watershed_boundary = ws.boundary['features'][0]['geometry']['coordinates']
                nwis_coordinates[site] = watershed_boundary
                error_value = 0

            except Exception as error_name:
                print('Try Again Coordinates: The error is with coordinates and error is ' + str(error_name))
                error_value += 1
                sleep(3)  # wait for 3 seconds before trying to fetch the data again

        sleep(5) 

    nwis_stats = pd.DataFrame(data=nwis_value_all, columns=nwis_index)

    nwis_stats.reset_index(drop=True, inplace=True)

    return nwis_stats, nwis_coordinates













'''def get_USGS_site_info(site_ids):

    NWISvalues_all = []
    str_error = 0
    print('Calculating NWIS streamflow id characteristics for ', len(site_ids), 'sites')

    pbar = ProgressBar()
    for site in pbar(site_ids):
        print('NWIS site: ', site)

            
        NWISinfo = nwis.get_record(sites=site, service='site')

        lat, lon = NWISinfo['dec_lat_va'][0], NWISinfo['dec_long_va'][0]

        # This sources the prestored data
        try:
            ws = streamstats.Watershed(lat=lat, lon=lon)
            print(ws)
        except:
            print('502 error, StreamStats down, using backup files')

        NWISindex = ['NWIS_site_id', 'Lat', 'Long', 'Drainage_area_mi2', 'Mean_Basin_Elev_ft', 'Perc_Forest',
                     'Perc_Develop',
                     'Perc_Imperv', 'Perc_Herbace', 'Perc_Slop_30', 'Mean_Ann_Precip_in']
        for i in range(5):
            try:

                print('Retrieving Drainage Area')

                darea = ws.get_characteristic('DRNAREA')['value']

                print('Retrieving Mean Catchment Elevation')

                elev = ws.get_characteristic('ELEV')['value']

                print('Retrieving Catchment Land Cover Information')

                forest = ws.get_characteristic('FOREST')['value']

                dev_area = ws.get_characteristic('LC11DEV')['value']



                imp_area = ws.get_characteristic('LC11IMP')['value']


                herb_area = ws.get_characteristic('LU92HRBN')['value']

                print('Retrieving Catchment Topographic Complexity')

                perc_slope = ws.get_characteristic('SLOP30_10M')['value']


                print('Retrieving Catchment Average Precip')

                precip = ws.get_characteristic('PRECIP')['value']


                NWISvalues = [site,
                              lat,
                              lon,
                              darea,
                              elev, forest,
                              dev_area,
                              imp_area,
                              herb_area,
                              perc_slope,
                              precip]

                print(NWISvalues)

                NWISvalues_all.append(NWISvalues)
                str_error = 0
            except Exception:
                
                if i != 5:
                    str_error = 1
                pass

            if str_error == 1:
                sleep(2)  # wait for 2 seconds before trying to fetch the data again
            else:
                break
        sleep(5) 

    NWIS_Stats = pd.DataFrame(data=NWISvalues_all, columns=NWISindex)

    NWIS_Stats = NWIS_Stats[colorder]

    NWIS_Stats.reset_index(drop=True, inplace=True)

    return NWIS_Stats'''
