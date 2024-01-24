from hydrotools.nwis_client.iv import IVDataService
import pandas as pd
import ulmo
import numpy as np
from hydrotools.nwm_client import utils

class Retriever:

    def get_usgs(site, start_date, end_date):
        service = IVDataService()
        usgs_data = service.get(
            sites=site,
            startDT= start_date,
            endDT= end_date
            )
        return usgs_data

    def get_snotel(sitecode, start_date, end_date):
      #  print(sitecode)

        #This is the latest CUAHSI API endpoint
        wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'

        #Daily SWE
        variablecode = 'SNOTEL:WTEQ_D'

        values_df = None
        try:
            #Request data from the server
            site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
            #print(site_values)

            #end_date=end_date.strftime('%Y-%m-%d')
            #Convert to a Pandas DataFrame   
            SNOTEL_SWE = pd.DataFrame.from_dict(site_values['values'])
            #Parse the datetime values to Pandas Timestamp objects
            SNOTEL_SWE['datetime'] = pd.to_datetime(SNOTEL_SWE['datetime'], utc=False)
            #Set the DataFrame index to the Timestamps
            SNOTEL_SWE = SNOTEL_SWE.set_index('datetime')
            #Convert values to float and replace -9999 nodata values with NaN
            SNOTEL_SWE['value'] = pd.to_numeric(SNOTEL_SWE['value']).replace(-9999, np.nan)
            #Remove any records flagged with lower quality
            SNOTEL_SWE = SNOTEL_SWE[SNOTEL_SWE['quality_control_level_code'] == '1']

            SNOTEL_SWE['station_id'] = sitecode
            #SNOTEL_SWE.index = SNOTEL_SWE.station_id
            SNOTEL_SWE = SNOTEL_SWE.rename(columns = {'value':f"{sitecode}_SWE(in)"})
            #col = [end_date]
            #SNOTEL_SWE = SNOTEL_SWE[col]#.iloc[-1:]


        except:
            print('Unable to fetch SWE data for site ', sitecode, 'SWE value: -9999')
            end_date=end_date.strftime('%Y-%m-%d')
            SNOTEL_SWE = pd.DataFrame(-9999, columns = ['station_id', end_date], index =[1])
            SNOTEL_SWE['station_id'] = sitecode
            SNOTEL_SWE = SNOTEL_SWE.set_index('station_id')


        return SNOTEL_SWE
    
    def valid_station(missing_value, data_number, last_year, states_list):
    
    
        modified_station = states_list[(states_list['year_number'] >= data_number) &
                                       (states_list['missing_value_percent'] <= missing_value) &
                                       (states_list.end_date.dt.year >= last_year)]
        return modified_station.reset_index(drop=True)

    # Function for getting NWM data
    def get_nhd_model_info(NWIS_sites):
        print('Getting NHD reaches')
       #Get NHD reach colocated with NWIS
        NHD_reaches = []
        NWIS_NHDPlus = pd.DataFrame(columns = ['NWISid','NHDPlusid'])

        for site in NWIS_sites.station.tolist():



            try:
                NHD_NWIS_df = utils.crosswalk(usgs_site_codes=site)
                NHD_segment = NHD_NWIS_df.nwm_feature_id.values[0]
                NHD_reaches.append(NHD_segment)

            except:
                NHD_segment = np.nan
                NHD_reaches.append(NHD_segment)

        NWIS_NHDPlus['NWISid'] = NWIS_sites.station
        NWIS_NHDPlus['state'] = NWIS_sites.state
        NWIS_NHDPlus['NHDPlusid'] = NHD_reaches

        return NHD_reaches, NWIS_NHDPlus

    def get_NWM_data(NWIS_NHDPlus, start, end, bucket):
        NWM_obs = {}
        reaches = NWIS_NHDPlus['NHDPlusid'].to_list()
        for reach_index, reach in enumerate(reaches):
    
            if reach > 1:
                r = str(int(reach))
                s = NWIS_NHDPlus.iloc[reach_index, 2]
                reachid = f"NWM_v2.1/NHD_segments_{s}.h5/NWM_{r}.csv"
                obj = bucket.Object(reachid)
                try:
                    body = obj.get()['Body']
                    NWM_obs[r] = pd.read_csv(body)
                    NWM_obs[r].rename(columns = {'Datetime':'datetime'}, inplace = True)
                    NWM_obs[r].set_index('datetime', inplace = True)
                    NWM_obs[r]=NWM_obs[r][start:end]
                    NWM_obs[r] = pd.DataFrame(NWM_obs[r]['NWM_flow'])
                    NWM_obs[r]['station_id'] =  NWIS_NHDPlus['NWISid'][ NWIS_NHDPlus['NHDPlusid'] == reach].values[0]
                    NWM_obs[r].reset_index(inplace = True)
                except:
                    NWM_obs[r] = -1000

            else:
                r = str(reach)
    
        return NWM_obs
