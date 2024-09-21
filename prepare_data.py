import xarray as xr
import os
import numpy as np
import pickle as pk
import pandas as pd

# User parameters :
dataset_path = os.path.join(r"/mnt/c",r"Users/romain.ribault/Documents/git_folders/saitec_demosath/2024-08-20_merged_June_results.nc")
df = xr.open_dataset(dataset_path)

# Define chanels for the dataset
channels = ['simu_AI_WindSpeed', 'simu_V_ST_TrueWindDir', 'simu_V_ST_TrueNacelleDir', 'simu_V_GridRealPowerLog', 'simu_V_MRU_Heave', 'simu_V_MRU_Pitch', 'simu_V_MRU_Heading'] # 'pitch', 'yaw']
envir_list = ['simu_hs', 'simu_tp', 'simu_dp',  'simu_theta10' ]
variable_list = channels

coordinate_list = list(df.coords)
not_drop_list = coordinate_list + variable_list + envir_list

# drop all variables not in variable_list
df = df.drop_vars([ var for var in df.variables if var not in not_drop_list] )
df = df.dropna(dim='time', how='any')

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

def record_data(step:str='train', start='2024-05-08 00:00:00', end= '2024-05-08 23:00:00', date_index_dict = {}) :
    global df
    dslice = df.sel(time=slice(start,end))
    data = get_numpy_input_2D_set(dslice, channels)
    date_index_dict[step] = get_pd_index(dslice)
    data = np.vstack(data)

    if step=='train' :
        scaler = [np.mean(data, axis=0), np.std(data, axis=0)]
        pickle_file_path = os.path.join('datasets/demosath', f'scaler.pkl')
        with open(pickle_file_path, 'wb') as f:
            pk.dump(scaler, f)

    pickle_file_path = os.path.join('datasets/demosath', f'{step}_set.pkl')
    with open(pickle_file_path, 'wb') as f:
        pk.dump(data, f)

    return date_index_dict

record_data_input =[
    {'step': 'train', 'start': '2024-05-08 00:00:00', 'end': '2024-05-08 23:00:00'},
    {'step': 'val', 'start': '2024-05-07 16:00:00', 'end': '2024-05-07 19:00:00'},
    {'step': 'test', 'start': '2024-05-07 20:00:00', 'end': '2024-05-07 23:00:00'},
]

date_index_dict={}
for input in record_data_input:
    date_index_dict = record_data(input['step'], input['start'], input['end'], date_index_dict )


pickle_file_path = os.path.join('datasets/demosath', f'timestamp.pkl')
with open(pickle_file_path, 'wb') as f:
    pk.dump(date_index_dict, f)


scaler = []




