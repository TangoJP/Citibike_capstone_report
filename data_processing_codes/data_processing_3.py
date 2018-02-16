# Create additional datasets to be used for classification tasks

import pandas as pd
import numpy as np
import os
import time
from data_process_methods import (add_date_information,
                            count_pickups_dropoffs,
                            add_station_information,
                            add_weather_information)

# Start timing the process
t0 = time.time()

file_dir = './data/downloaded/'
file_year = ['2013', '2014', '2015', '2016', '2017']
file_months = ['07', '08', '09', '10', '11', '12']
                #['01', '02', '03', '04', '05', '06']

file_end = '_citibike-tripdata.csv'

counts = np.zeros(4)
total_iteration = len(file_year) * len(file_months)
iteration = 1

for i in range(len(file_year)):
    # Set up a dataframe to contain a yearly data
    yearly_data = pd.DataFrame()

    # Iterate through months
    for j in range(len(file_months)):
        # Import monthly data
        file_name = file_dir + file_year[i] + file_months[j] + file_end
        # Ignore if the month's data non-existent
        if os.path.isfile(file_name) == False:
            print(file_year[i] + file_months[j] + file_end + ' does not exist')
            iteration += 1
        else:
            # Read in the monthly data
            data = pd.read_csv(file_name)
            print('%.1fs: Processing %d/%d, %s...' \
                         % ((time.time() - t0),
                            iteration,
                            total_iteration,
                            (file_year[i] + file_months[j] + file_end)))

            # Add additional datetime information
            print('\tAdding date information...')
            data = add_date_information(data)

            # Append the monthly data to the yearly_data
            print('\tAppending data...')
            yearly_data = pd.concat([yearly_data, data])

            iteration += 1

    # Save combined data for each year
    print('Saving the file...')
    savepath = './data/processed/yearly/' + file_year[i]
    yearly_data.to_csv(savepath + '_2ndHalf_data.csv')

print('...Processing complete')
