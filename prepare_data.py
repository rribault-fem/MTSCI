import xarray as xr
import os
import numpy as np
import pickle as pk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
import logging






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

def record_data(step:str='train', start='2024-03-08 00:00:00', end= '2024-05-08 23:00:00', date_index_dict = {}) :
    global df
    dslice = df.sel(time=slice(start,end))
    data = get_numpy_input_2D_set(dslice, channels)
    date_index_dict[step] = get_pd_index(dslice)
    data = np.vstack(data)

    skicit_scaler_file_path = os.path.join('datasets/demosath', f'skicit_scaler.pkl')

    if step=='train' :
        skicit_scaler = MinMaxScaler()
        skicit_scaler.fit(data)

        with open(skicit_scaler_file_path, 'wb') as f:
            pk.dump(skicit_scaler, f)

    
    with open(skicit_scaler_file_path, "rb") as fb:
        skicit_scaler = pk.load(fb)


    pickle_file_path = os.path.join('datasets/demosath', f'{step}_set.pkl')
    with open(pickle_file_path, 'wb') as f:
        data_scaled = skicit_scaler.transform(data)
        pk.dump(data_scaled, f)

    if step == 'train' :
        scaler = [np.mean(data_scaled, axis=0), np.std(data_scaled, axis=0)]
        pickle_file_path = os.path.join('datasets/demosath', f'scaler.pkl')
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
    dataset_path = os.path.join(r"~",r"git_folders/torchydra/2024-07-02_merged_simu_sensors_db_saved.nc")
    

    # Define chanels for the dataset
    channels = ['simu_AI_WindSpeed', 'simu_V_ST_TrueWindDir', 'simu_V_ST_TrueNacelleDir', 'simu_V_GridRealPowerLog', 'simu_V_MRU_Heave', 'simu_V_MRU_Pitch', 'simu_V_MRU_Roll', 'simu_V_MRU_Longitude_rel', 'simu_V_MRU_Latitude_rel', 'simu_V_RotorRpm', 'simu_V_MRU_Heading', 'simu_V_SPM_LOAD_Pin_1', 'simu_V_SPM_LOAD_Pin_2', 'simu_V_SPM_LOAD_Pin_3', 'simu_V_SPM_LOAD_Pin_4', 'simu_V_SPM_LOAD_Pin_5', 'simu_V_SPM_LOAD_Pin_6'] # 'pitch', 'yaw']
    envir_list = ['simu_hs', 'simu_tp', 'simu_dp',  'simu_theta10' ]

    # channels which requires a cos / sin decomposition to avoid 360-->0 variations.
    heading_angle_vars = ['simu_V_ST_TrueWindDir', 'simu_V_ST_TrueNacelleDir','simu_V_MRU_Heading']

    # How to split train / val / test sets.
    record_data_input =[
        {'step': 'train', 'start': '2023-12-01 00:00:00', 'end': '2024-05-08 13:00:00'},
        {'step': 'val', 'start': '2024-05-08 16:00:00', 'end': '2024-05-08 18:00:00'},
        # {'step': 'test', 'start': '2024-05-08 14:00:00', 'end': '2024-05-30 23:00:00'},
        {'step': 'test', 'start': '2024-05-08 16:00:00', 'end': '2024-05-08 17:00:00'}
    ]

    df = xr.open_dataset(dataset_path)
    variable_list = channels
    coordinate_list = list(df.coords)
    not_drop_list = coordinate_list + variable_list + envir_list
    # drop all variables not in variable_list
    df = df.drop_vars([ var for var in df.variables if var not in not_drop_list] )
    df = df.dropna(dim='time', how='any')

    df, new_vars = get_cos_sin_from_angle(heading_angle_vars, df)
    for drop_var in heading_angle_vars:
        channels.remove(drop_var)
    for new_var in new_vars:
        channels.insert(8, new_var)

    # write the channel list to text file for logging
    with open(os.path.join('datasets/demosath','channels.txt'), 'w') as f:
            f.write(str(channels))

    date_index_dict={}
    for input in record_data_input:
        date_index_dict = record_data(input['step'], input['start'], input['end'], date_index_dict )


    pickle_file_path = os.path.join('datasets/demosath', f'timestamp.pkl')
    with open(pickle_file_path, 'wb') as f:
        pk.dump(date_index_dict, f)





