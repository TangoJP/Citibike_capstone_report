# Combine the pre-processed yearly data with extended information
# to create a whole datasets. Bike station-related information and
# the weather information are appended to the ridership data

import pandas as pd
import numpy as np
import os
import time
from data_process_methods import add_extra_information

# Start timing the process
t0 = time.time()

file_dir = './data/processed/test2/'
file_year = ['2013', '2014', '2015', '2016', '2017']
file_names = ['p_s', 'p_ns', 'd_s', 'd_ns']

new_dir = './data/processed/extended2/'

# iterate over file names
for file_name in file_names:
    if 'p_' in file_name:
        label = 'pickups'
    else:
        label = 'dropoffs'

    data = pd.DataFrame(columns=['date', 'st_id', label])
    for i in range(len(file_year)):
        file_path = file_dir + file_year[i] + '_' + file_name + '.csv'
        addition = pd.read_csv(file_path)
        data = pd.concat([data, addition])
    data.to_csv(new_dir + file_name + '_alltime.csv', index=False)

# Import extra information data that were separately prepared
st_info = './data/processed/stations_info_extended_complete.csv'
weather = './data/processed/ny_weather.csv'

# Append the extra information and save
for file_name in file_names:
    file_path = new_dir + file_name + '_alltime.csv'
    ext_data = pd.read_csv(file_path)
    print('%s: Length before %d' % (file_name, len(ext_data)))
    ext_data = add_extra_information(ext_data, st_info, weather)
    print('%s: Length after %d' % (file_name, len(ext_data)))
    ext_data.to_csv(new_dir + file_name + '_alltime_extended.csv', index=False)
