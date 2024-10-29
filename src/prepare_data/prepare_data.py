import xarray as xr
import os
import numpy as np
import pickle as pk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
import logging
from validity_domain import ValidityDomain

def get_cos_sin_from_angle(angles:List[str], df:xr.Dataset):
    new_variables = []
    print('###')
    print(f'get cos and sin decomposition of the data {angles}')
    for angle in angles :
        print(f'get cos and sin decomposition of the data {angle}')

        df_cos = get_xr_dataset_time_time_sensor(angle+'_cos', np.cos(2*np.pi*df[angle].values/365), f'cos of {angle} values', df.time.values, dims=list(df[angle].coords))
        df_sin = get_xr_dataset_time_time_sensor(angle+'_sin', np.sin(2*np.pi*df[angle].values/365), f'sin of {angle} values', df.time.values, dims=list(df[angle].coords))
        
        df = xr.merge([df, df_cos, df_sin], compat = 'no_conflicts')
        new_variables += [angle+'_cos', angle+'_sin']

    return df, new_variables

def get_xr_dataset_time_time_sensor(array_name : str, array_values : np.array, description : str , time_values : np.array, dims : List[str] = ["time", 'time_sensor']) :
    
    data_array_dict = {
            "attrs":{
                "description" : description},
            "dims" : dims,
            "data" : array_values,
            "name" : array_name
        }

    DataArray  =  xr.DataArray.from_dict(data_array_dict)
    Dataset = xr.Dataset(
        { array_name : DataArray}
    )

    return Dataset

def get_numpy_input_2D_set(df, channels) :
    # loading channels data in numpy for CNN 

    input_channel_set = np.empty_like(np.expand_dims(df[channels[0]].values, axis = 2))
    for channel in channels :
        input_channel = np.expand_dims(df[channel].values, axis = 2)
        input_channel_set = np.append(input_channel_set, input_channel, axis = 2)
    input_channel_set = np.delete(input_channel_set, 0, axis=2)
    return input_channel_set

def get_pd_index(df):
    date_index=[]
    for time in df['time']:
        global_time = df.sel(time = time)['time'].values + pd.to_timedelta(df.sel(time = time)['time_sensor'], unit = 's').round('ms').values
        date_index.append(global_time)
    date_index = np.concatenate(date_index)   
    return date_index

def record_data(step:str='train', dslice : xr.Dataset=None , date_index_dict = {}) :

    # rolling mean to resample at 1Hz
    # dslice = dslice.rolling(time_sensor=5).mean().dropna("time_sensor")
    data = get_numpy_input_2D_set(dslice, channels)
    # from 5Hz to 1Hz
    # data = data[:,::5,:]

    date_index_dict[step] = get_pd_index(dslice)
    # date_index_dict[step] = date_index_dict[step][::5]

    data = np.vstack(data)

    skicit_scaler_file_path = os.path.join(output_dir, f'skicit_scaler.pkl')

    if step=='train' :
        skicit_scaler = MinMaxScaler()
        skicit_scaler.fit(data)

        # User defined scaler to avoid gradient skyrocket gradients during training
        with open(skicit_scaler_file_path, 'wb') as f:
            pk.dump(skicit_scaler, f)

    
    with open(skicit_scaler_file_path, "rb") as fb:
        skicit_scaler = pk.load(fb)


    pickle_file_path = os.path.join(output_dir, f'{step}_set.pkl')
    with open(pickle_file_path, 'wb') as f:
        # Data are scaled before being saved.
        data_scaled = skicit_scaler.transform(data)
        pk.dump(data_scaled, f)

    # Specific Scaling data for MTSCI network
    if step == 'train' :
        scaler = [np.mean(data_scaled, axis=0), np.std(data_scaled, axis=0)]
        pickle_file_path = os.path.join(output_dir, f'scaler.pkl')
        with open(pickle_file_path, 'wb') as f:
            pk.dump(scaler, f)

    return date_index_dict


def from_sin_cos_to_heading(head_sin: np.array, head_cos: np.array) -> np.array:

    # Compute the wind direction in radians
    heading_rad = np.arctan2(head_sin, head_cos)

    # Convert the wind direction from radians to degrees
    heading_deg = np.degrees(heading_rad)

    # Ensure the wind direction is in the range [0, 360)
    heading_deg = (heading_deg + 360) % 360

    return heading_deg


if __name__ == '__main__' :

    # User parameters :
    dataset_path = os.path.join(r"~",r"git_folder/torchydra/2024-07-02_merged_simu_sensors_db_saved.nc")
    output_dir = 'datasets/demosath_3'

    # Define chanels for the dataset
    channels = ['simu_AI_WindSpeed', 'simu_V_ST_TrueWindDir', 'simu_V_ST_TrueNacelleDir', 'simu_V_GridRealPowerLog', 'simu_V_MRU_Heave', 'simu_V_MRU_Pitch', 'simu_V_MRU_Roll', 'simu_V_MRU_Longitude_rel', 'simu_V_MRU_Latitude_rel', 'simu_V_RotorRpm', 'simu_V_MRU_Heading', 'simu_V_SPM_LOAD_Pin_1', 'simu_V_SPM_LOAD_Pin_2', 'simu_V_SPM_LOAD_Pin_3', 'simu_V_SPM_LOAD_Pin_4', 'simu_V_SPM_LOAD_Pin_5', 'simu_V_SPM_LOAD_Pin_6'] # 'pitch', 'yaw']
    envir_list = ['simu_hs', 'simu_tp', 'simu_dp',  'simu_theta10', 'simu_mag10' ]

    # channels which requires a cos / sin decomposition to avoid 360-->0 variations.
    heading_angle_vars = ['simu_V_ST_TrueWindDir', 'simu_V_ST_TrueNacelleDir','simu_V_MRU_Heading']

    # How to split train / val / test sets.
    # record_data_input =[
    #     {'step': 'train', 'start': '2024-01-01 00:00:00', 'end': '2024-05-08 16:00:00'},
    #     {'step': 'val', 'start': '2024-05-08 16:00:00', 'end': '2024-05-08 18:00:00'},
    #     # {'step': 'test', 'start': '2024-05-08 14:00:00', 'end': '2024-05-30 23:00:00'},
    #     {'step': 'test', 'start': '2024-05-08 16:00:00', 'end': '2024-05-08 23:00:00'}
    # ]

    df = xr.open_dataset(dataset_path)
    variable_list = channels
    coordinate_list = list(df.coords)
    not_drop_list = coordinate_list + variable_list + envir_list
    # drop all variables not in variable_list
    df = df.drop_vars([ var for var in df.variables if var not in not_drop_list] )
    df = df.dropna(dim='time', how='any')

    # Drop Heave higher than 6m
    df = df.where(df['simu_V_MRU_Heave'].max(dim='time_sensor') <4, drop=True)

    # Drop when the turbine is producing
    df = df.where(df['simu_V_RotorRpm'].mean(dim='time_sensor') <3, drop=True)

    for i in range(1,7):
        df = df.where(df[f'simu_V_SPM_LOAD_Pin_{i}'].max(dim='time_sensor')<100000, drop=True)

    df, new_vars = get_cos_sin_from_angle(heading_angle_vars, df)

    drop_filter_vars = ['simu_V_GridRealPowerLog', 'simu_V_RotorRpm']

    for drop_var in heading_angle_vars + drop_filter_vars:
        channels.remove(drop_var)
    for new_var in new_vars:
        channels.insert(1, new_var) # insert new_vars at the same place as in df

    # write the channel list to text file for logging
    with open(os.path.join(output_dir,'channels.txt'), 'w') as f:
            f.write(str(channels))

    val_domain = ValidityDomain()

    df_train, df_test = val_domain.find_test_set_in_model_validity_domain(df)

    date_index_dict={}
    date_index_dict = record_data('train', df_train, date_index_dict)
    date_index_dict = record_data('val', df_test.isel(time=slice(0,2)), date_index_dict)
    date_index_dict = record_data('test', df_test, date_index_dict)

    pickle_file_path = os.path.join(output_dir, f'timestamp.pkl')
    with open(pickle_file_path, 'wb') as f:
        pk.dump(date_index_dict, f)





