#%%

# Bsic packages
from progressbar import ProgressBar
from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pyarrow.parquet as pq
import pyarrow as pa

# Hydrological Packages
from scipy import optimize
#import ee
import ulmo
import streamstats
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, GeometryCollection
from shapely import wkb
from shapely.wkt import loads as load_wkt
from pyproj import Transformer

# System package
import platform
import os
import time
from datetime import datetime

# My packages
from g_data_get import Retriever
import g_Get_StreamStats
#import EE_funcs

# AWS packages
import boto3

import xarray as xr

#%% Identify Paths
current_folder_path = Path.cwd()
home_folder = current_folder_path.parents[2]
data_folder = os.path.join(home_folder, 'Data', 'input')
result_folder = os.path.join(home_folder, 'Data', 'Processed')


#%% Identify Dates
start = '1990-01-01'  # Start date.
end = '2020-12-31'  # End date.
end_year = 2020  # Prefered end date. 
missing_data = 10  # Prefered missing data value. 
time_period = 11  # Prefered number of years. 

#%% Start AWS S3

# Start the S3 bucket access. 
from botocore.client import Config
from botocore import UNSIGNED
#load access key
home = os.path.expanduser('~')
keypath = "04.personal_files/aws_key.csv"
access = pd.read_csv(f"{home}/{keypath}")

#start session
session = boto3.Session(
    aws_access_key_id=access['Access key ID'][0],
    aws_secret_access_key=access['Secret access key'][0])
bucket_name = 'streamflow-app-data'
s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket(bucket_name)

#%% Get Stations List

def data_stations(path=None):

    all_stations = gpd.read_parquet(path) 
    all_stations.rename(columns={'NWIS_site_id': 'station_id'}, inplace=True)
    all_stations.station = all_stations.station_id.astype(int).astype(str)
    all_stations = all_stations.to_crs("EPSG:4326")

    stations = ['10134500',
    '10141000',
    '10105900',
    '10136500',
    '10132000',
    '10128500',
    '10155500',
    '10171000',
    '10145400',
    '10132500',
    '10155000',
    '10126000',
    '10166430',
    '10130500',
    '10168000',
    '10011500',
    '10131000',
    '10109000',
    '10113500',
    '10129500',
    '10137500',
    '10154200',
    '10150500',
    '10146000',
    '10140100',
    '10156000',
    '10146400',
    '10092700',
    '10039500',
    '10068500']
    accpeted_stations = all_stations[all_stations['station_id'].isin(stations)].reset_index(drop=True)
    return accpeted_stations

accpeted_stations = data_stations(path='/home/shared/NWM_ML/forcings/utah_files/stations_with_stats.parquet')

#%% Modify Land Use Data
def add_land_use(data):
    data['imperv_perc'] = data['veg_type_1_perc']

    data['agri_perc'] = data['veg_type_2_perc'] + data['veg_type_3_perc'] + data['veg_type_4_perc'] \
    + data['veg_type_5_perc'] + data['veg_type_6_perc']

    data['forest_perc'] = 0

    for ii in range(7, 15):
        data['forest_perc'] = data['forest_perc'] + data[f'veg_type_{ii}_perc'] 

    data = data[['station_id', 'state', 'geometry', 'dec_lat_va', 'dec_long_va', 'total_area', 'basin_mean_elevation', 'basin_mean_slope', 'imperv_perc', 'agri_perc', 'forest_perc']]

    return data

accpeted_stations = add_land_use(accpeted_stations)

#%% Get NHDPlus IDs

def add_nhdplusid(data):

    NWIS_NHDPlus = Retriever.get_nhd_model_info(data)[1]  # Get NWM reaches for each USGS station
    NWIS_NHDPlus.dropna(inplace=True)  # Drop the stations that don't have NWM reaches. 
    NWIS_NHDPlus.reset_index(drop=True, inplace=True)
    NWIS_NHDPlus.rename(columns={'NWISid': 'station_id'}, inplace=True)
    data = pd.merge(data, NWIS_NHDPlus.iloc[:, :-1], on='station_id')


    return data


accpeted_stations = add_nhdplusid(accpeted_stations)

#%% Get NWM Data


def data_nwm(start=None, end=None, data=None, online=False, path=None, bucket_path=None):

    if online == True:
        # Get data for each reach
        NWM_obs = Retriever.get_NWM_data(data, start, end, bucket)

        # Merge into one DF
        NWM_obs_df = pd.DataFrame()
        pbar = ProgressBar()
        for site in tqdm(NWM_obs):

            # This section is to make sure to remove the USGS gauges that don't have any NWM reach. 
            if isinstance(NWM_obs[site], int):
                print('Stations without NWM data:')
                print(site)
                NWM_obs[site] = pd.DataFrame({'NWM_flow': [np.nan] * 11231, 'datetime': NWM_obs['7887898']['datetime'], 'station_id': [np.nan] * 11231})
            NWM_obs_df = pd.concat([NWM_obs[site], NWM_obs_df])

        # Convert to datetime to make sure it joins nicely with the clim_df
        NWM_obs_df['datetime'] = pd.to_datetime(NWM_obs_df['datetime'])
        NWM_obs_df['NWM_flow'] = NWM_obs_df['NWM_flow'] * 0.028316831998814504 

        table = pa.Table.from_pandas(NWM_obs_df)
        pq.write_table(
            table, 
            path,
            compression='brotli')

    else:
        NWM_obs_df = pd.read_parquet(path)


    data = data[data['station_id'].isin(NWM_obs_df.dropna().station_id.unique())].reset_index(drop=True)

    return NWM_obs_df, data

NWM_obs_df, accpeted_stations = data_nwm(start=start, end=end, data=accpeted_stations, online=True, path=f'{result_folder}/data_nwm.parquet', bucket_path=bucket)


#%% Get USGS Data for the Valid Stations

def data_usgs(stations=None, online=False, path=None):
    if online == True:
        # Create variables
        flow = {}
        all_sites_flow = pd.DataFrame()
        pbar = ProgressBar()
        # Run the code for each station
        for site in tqdm(stations.values.flatten().tolist()):
            
            #print('Getting streamflow data for ', site, end='\r', flush=True)
            flow[site] = Retriever.get_usgs(site, start, end)

            #make the date the index for plotting and resampling
            flow[site]['datetime'] = flow[site]['value_time']
            flow[site].set_index('datetime', inplace = True)

            #clean the data
            flow[site] = flow[site][flow[site]['value'] > 0]

            #resample to a daily resolution
            flow[site] = pd.DataFrame(flow[site]['value']).rename(columns = {'value':'flow_cfs'})
            flow[site] = flow[site].resample('D').mean()
            flow[site]['flow_cms'] = flow[site]['flow_cfs'] * 0.028316831998814504
            flow[site]['station_id'] = site
            flow[site].reset_index(inplace=True)
            flow[site].drop(columns=['flow_cfs'], inplace=True)


            all_sites_flow = pd.concat([all_sites_flow, flow[site]])
        all_sites_flow.reset_index(drop=True, inplace=True)

        table = pa.Table.from_pandas(all_sites_flow)
        pq.write_table(
            table, 
            path,
            compression='brotli')

    else:
        all_sites_flow = pd.read_parquet(path)
    
    
    return all_sites_flow

all_sites_flow = data_usgs(stations=accpeted_stations.station_id, online=False, path=f'{result_folder}/data_streamflow.parquet')

#%%

def add_properties(data=None, online=False, path=None):

    if online == True:

        # Extract the day of the year
        data['day_of_year'] = data['datetime'].dt.dayofyear

        # Group by station and day of the year, then calculate min, max, mean, and median
        aggregated_data = data.groupby(['station_id', 'day_of_year'])['flow_cms'].agg(['min', 'max', 'mean', 'median']).reset_index()

        # Merge the aggregated data back into the original DataFrame
        modified_data = data.merge(aggregated_data, on=['station_id', 'day_of_year'], how='left', suffixes=('', '_agg'))

        table = pa.Table.from_pandas(modified_data)
        pq.write_table(
            table, 
            path,
            compression='brotli')

    else:
        modified_data = pd.read_parquet(path)

    return modified_data

modified_flow = add_properties(all_sites_flow, online=False, path=f'{result_folder}/data_streamflow.parquet')

#%% Get Storage Data

def data_storage(storage_info=None, boundry_list=None, online=False, storage_path=None, read_path=None):
    
    if online == True:
        # Create a GeoDataFrame with storage coordiates. 
        geometry_points = [Point(xy) for xy in zip(storage_info['LONG'], storage_info['LAT'])]
        points_gdf = gpd.GeoDataFrame(geometry=geometry_points, crs='EPSG:4326')

        # Create the variables.
        all_storage = pd.DataFrame()

        all_station_dict = {}
        for i in tqdm(range(len(boundry_list))):

            # Search to see which storages are in the watershed. 
            snotel_station_list = storage_info[points_gdf.within(boundry_list.loc[[i],'geometry'].geometry.iloc[0])].reset_index(drop=True)
            all_station_dict[boundry_list.iloc[i, 0]] = snotel_station_list

            
            # Create variables
            all_snotel_swe = pd.DataFrame()
            
            # Check the number of storages for each station. 
            if len(snotel_station_list) >= 1:
                for station_index in range(len(snotel_station_list)):
                    sitecode = snotel_station_list.iloc[station_index, 0]  # Get the storage name. 

                    # Read the specific storage data, and prepare it for the next steps. 
                    storage_data = pd.read_csv(f'{storage_path}_{sitecode}.csv')
                    storage_data['date'] = pd.to_datetime(storage_data['date'])  # Convert the type of a column..
                    # snotel_swe = storage_data[(storage_data['storage_name'] == sitecode) & (storage_data['date'] >= start) & (storage_data['date'] <= end)]
                    snotel_swe = storage_data[(storage_data['date'] >= start) & (storage_data['date'] <= end)]

                    # snotel_swe.drop([snotel_swe.columns[0], 'storage_name'], inplace=True, axis=1) 
                    snotel_swe = snotel_swe[['date', 'storage']].reset_index(drop=True)

                    # Get the percent of storage capacity which is full. 
                    snotel_swe['storage'] = snotel_swe['storage'].fillna(method='ffill')
                    snotel_swe['storage'] = snotel_swe['storage'] / snotel_station_list.loc[[station_index], 'capacity(mcm)'].values * 100

                    snotel_swe.rename(columns={'storage': sitecode}, inplace=True)

                    # If it is the first snow station.
                    if station_index == 0:
                        all_snotel_swe = snotel_swe.copy()
                        
                    # If it is not the first snow station. 
                    else:
                        all_snotel_swe = pd.merge(all_snotel_swe, snotel_swe, on='date')  # Merge based on date. 


                # Prepare data. 
                all_snotel_swe.set_index('date', inplace=True)
                all_snotel_swe['sum'] = all_snotel_swe.sum(axis=1)/ len(snotel_station_list)  # Average of all stations. 


                all_snotel_swe.drop(all_snotel_swe.iloc[:, :-1], inplace=True, axis=1)
                all_snotel_swe.rename(columns={'sum': 'storage'}, inplace=True)
                all_snotel_swe['station_id'] = boundry_list.iloc[i, 0]

            else:
                
                # Create an zero dataframe. 
                all_snotel_swe = pd.DataFrame({'storage': [0], 'station_id': boundry_list.iloc[i, 0]}, index=pd.date_range(start=start, end=end))

            # Add all the data for each station to the previous one and make a complete dataset.
            all_storage = pd.concat([all_storage, all_snotel_swe])
        all_storage.reset_index(inplace=True) 
        all_storage.rename(columns={'index': 'datetime'}, inplace=True)
        all_storage['datetime'] = pd.to_datetime(all_storage['datetime'])
        all_storage['station_id'] = all_storage['station_id'].astype(int).astype(str)
        table = pa.Table.from_pandas(all_storage)
        pq.write_table(
            table, 
            read_path,
            compression='brotli')

    else: 

        all_storage = pd.read_parquet(read_path)

    return all_storage


storage_list = pd.read_parquet(f'{data_folder}/final_reservoir_info.parquet')

all_storage = data_storage(storage_info=storage_list, boundry_list=accpeted_stations, online=False, storage_path=f'{data_folder}/reservoir_data/ResOpsUS', read_path=f'{result_folder}/data_storage.parquet')

#%% Get SNOTEL Data


def data_swe(start=None, end=None, snotel_info=None, boundry_list=None, online=False, read_path=None):

    if online == True:
        # Convert start and end to datetime objects
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        
        # Calculate the number of days between the two dates
        days_between = (end_date - start_date).days
        geometry_points = [Point(xy) for xy in zip(snotel_info['longitude'], snotel_info['latitude'])]
        snotel_gdf = gpd.GeoDataFrame(geometry=geometry_points, crs='EPSG:4326')
        all_snow_swe = pd.DataFrame()

        all_station_dict = {}
        for i in tqdm(range(len(accpeted_stations))):

            snotel_station_list = snotel_info[snotel_gdf.within(boundry_list.loc[[i],'geometry'].geometry.iloc[0])].reset_index(drop=True)
            all_station_dict[boundry_list.iloc[i, 0]] = snotel_station_list

            all_snotel_swe = pd.DataFrame()
            all_snotel_swe = None  # Initialize all_snotel_swe as None
            first_valid_station = False  # Initialize a flag to check if the first valid station has been set
            if len(snotel_station_list) >= 1:
                for station_index in range(len(snotel_station_list)):
                    sitecode = snotel_station_list.iloc[station_index, 0]
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)

                    snotel_swe = Retriever.get_snotel(sitecode, start_date, end_date)

                    snotel_swe.drop(snotel_swe.iloc[:, 1:], inplace=True, axis=1)
                    

                    if len(snotel_swe) / days_between * 100 < 90:
                        continue

                    if not first_valid_station:
                        all_snotel_swe = snotel_swe.copy()  # Set the first valid station data
                        first_valid_station = True  # Mark the first valid station as set
                    else:
                        all_snotel_swe = pd.merge(all_snotel_swe, snotel_swe, left_index=True, right_index=True)
                all_snotel_swe['sum'] = all_snotel_swe.sum(axis=1)/ len(snotel_station_list)
                all_snotel_swe.drop(all_snotel_swe.iloc[:, :-1], inplace=True, axis=1)
                all_snotel_swe.rename(columns={'sum': 'swe'}, inplace=True)
                all_snotel_swe['station_id'] = boundry_list.iloc[i, 0]

            else:

                all_snotel_swe = pd.DataFrame({'swe': [0], 'station_id': boundry_list.iloc[i, 0]}, index=pd.date_range(start=start, end=end))
            all_snow_swe = pd.concat([all_snow_swe, all_snotel_swe])
        all_snow_swe = all_snow_swe.iloc[:, :2]
        all_snow_swe.reset_index(inplace=True)
        all_snow_swe.rename(columns={'index': 'datetime'}, inplace=True)
        all_snow_swe['datetime'] = pd.to_datetime(all_snow_swe['datetime'])
        all_snow_swe['station_id'] = all_snow_swe['station_id'].astype(int).astype(str)
        table = pa.Table.from_pandas(all_snow_swe)
        pq.write_table(
            table, 
            read_path,
            compression='brotli')

    else:

        all_snow_swe = pd.read_parquet(read_path)

    return all_snow_swe

snotel_df = pd.read_csv(f'{data_folder}/Snotel-CDEC-SnowObs.csv')

all_snow_swe = data_swe(start=start, end=end, snotel_info=snotel_df, boundry_list=accpeted_stations, online=False, read_path=f'{result_folder}/data_swe.paqruet')



# %% Get Climate Data

def add_climate(start=None, end=None, online=False, station_list=None, read_path=None, write_path=None):
    
    if online == True:
        # Open the Zarr dataset
        ds_all = xr.open_zarr(read_path)
        
        # Filter the dataset for the specified stations and time range
        filtered_ds = ds_all.sel(station=station_list, time=slice(start, end))
        
        # Convert the dataset to a pandas DataFrame and reset the index
        filtered_df = filtered_ds.to_dataframe().reset_index()
        
        # Rename the columns to more descriptive names
        filtered_df.rename(columns={'station': 'station_id', 'time': 'datetime'}, inplace=True)
        
        # Set 'datetime' as the index along with 'station_id'
        filtered_df.set_index(['datetime', 'station_id'], inplace=True)

        # Initialize an empty DataFrame to store the results
        resampled_df = pd.DataFrame()

        # Resample data for each station separately
        for station in filtered_df.index.get_level_values('station_id').unique():
            station_df = filtered_df.xs(station, level='station_id')
            
            # Resample the data
            temp_resampled = station_df['temperature'].resample('D').mean() - 273.15
            precip_resampled = station_df['precipitation'].resample('D').sum() * 24 * 3600
            
            # Create a DataFrame for the resampled data
            station_resampled_df = pd.DataFrame({
                'station_id': station,
                'datetime': precip_resampled.index,
                'precipitation': precip_resampled.values,
                'temperature': temp_resampled.values
            })
            
            # Append to the main DataFrame
            resampled_df = pd.concat([resampled_df, station_resampled_df], ignore_index=True)

        table = pa.Table.from_pandas(resampled_df)
        pq.write_table(
            table, 
            write_path,
            compression='brotli')
    else:
        resampled_df = pd.read_parquet(write_path)

    return resampled_df


data_climate = add_climate(start=start, end=end, online=True, station_list=accpeted_stations.station_id.unique(), read_path='/home/shared/NWM_ML/climate_data.zarr', write_path=f'{result_folder}/data_climate.paqruet')


# %% Create Final Database
seasonality = pd.read_csv(f'{data_folder}/seasonality.csv')


variable_list = [data_climate, all_storage, all_snow_swe, NWM_obs_df]

final_data_set = pd.merge(accpeted_stations, modified_flow, on=['station_id'])
final_data_set['month'] = final_data_set.datetime.dt.month
final_data_set = pd.merge(final_data_set, seasonality, on=['month'])
for variable in variable_list:
    final_data_set = pd.merge(final_data_set, variable, on=['datetime', 'station_id'])
final_data_set = final_data_set.drop(columns=['month', 'geometry'])
final_data_set = final_data_set[['station_id', 'NHDPlusid', 'datetime', 'state', 'dec_lat_va', 'dec_long_va',
       'total_area', 'basin_mean_elevation', 'basin_mean_slope', 'imperv_perc',
       'agri_perc', 'forest_perc',
       'day_of_year', 'min', 'max', 'mean', 'median', 's1', 's2',
       'precipitation', 'temperature', 'storage', 'swe', 'NWM_flow', 'flow_cms']]
final_data_set.dropna(inplace=True)

table = pa.Table.from_pandas(final_data_set)
pq.write_table(
    table, 
    f'{result_folder}/data_model_input.parquet',
    compression='brotli')


# %%